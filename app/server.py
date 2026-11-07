"""
Inference server for KyotoLM.

Run with:
    uvicorn app.server:app --host 0.0.0.0 --port 8000

Set KYOTOLM_CHECKPOINT to override which checkpoint gets loaded
(defaults to ~/Downloads/checkpoints/sft.pt).
"""
import os
import time
import uuid
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tokenizers import Tokenizer

from models import MODELS

TOKENIZER_PATH = "tokenizer.json"
DEFAULT_CHECKPOINT = "~/Downloads/checkpoints/sft.pt"
CHECKPOINT_PATH = Path(os.environ.get("KYOTOLM_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser()

app = FastAPI(title="KyotoLM Inference Server")

# model.generate() mutates shared RoPE-cache buffers keyed on the current sequence
# length — concurrent requests at different lengths would race and corrupt each
# other's output, so all generation is serialized through this lock.
_generate_lock = Lock()


def _load_model(checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    d = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tc = d["config"]
    arch = MODELS[tc.get("model", "v2")]
    model = arch.KyotoLM(
        arch.Config(
            vocab_size=d["vocab_size"],
            n_embedding_dim=tc.get("n_embedding_dim", tc.get("embedding_dim", 768)),
            n_head=tc.get("n_heads", 6),
            n_layers=tc.get("n_layers", 12),
        )
    )
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in d["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


model, device = _load_model(CHECKPOINT_PATH)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")


def _format_chat_prompt(messages: list[dict]) -> list[int]:
    """Mirrors the SFT training format from data_loader.py's SFTDataset."""
    token_ids = [bos_id]
    for turn in messages:
        role, content = turn["role"], turn["content"]
        if role == "system":
            token_ids.extend(tokenizer.encode(f"System: {content}\n\n").ids)
        elif role == "user":
            token_ids.extend(tokenizer.encode(f"User: {content}\n\nAssistant: ").ids)
        elif role == "assistant":
            token_ids.extend(tokenizer.encode(content).ids)
            token_ids.append(eos_id)
        else:
            raise HTTPException(400, f"Unknown role: {role}")
    return token_ids


def _run_generate(token_ids: list[int], max_tokens: int, temperature: float, top_k: int, repetition_penalty: float):
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    with _generate_lock, torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token=eos_id,
            repetition_penalty=repetition_penalty,
        )
    generated_ids = out[0].tolist()[len(token_ids):]
    finish_reason = "stop"
    if eos_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(eos_id)]
    else:
        finish_reason = "length"
    return tokenizer.decode(generated_ids).strip(), finish_reason


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.3


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "kyotolm"
    choices: list[ChatCompletionChoice]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.3


class GenerateResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "checkpoint": str(CHECKPOINT_PATH)}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    token_ids = _format_chat_prompt([m.model_dump() for m in req.messages])
    text, finish_reason = _run_generate(token_ids, req.max_tokens, req.temperature, req.top_k, req.repetition_penalty)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    token_ids = tokenizer.encode(req.prompt).ids
    text, _ = _run_generate(token_ids, req.max_tokens, req.temperature, req.top_k, req.repetition_penalty)
    return GenerateResponse(text=text)


# Mounted last so it doesn't shadow the API routes above — serves the chat UI at "/".
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
