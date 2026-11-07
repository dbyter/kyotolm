from models import kyotolmv2, kyotolmv4

# Checkpoints record their arch under config["model"]; older ones predate the key and are v2.
MODELS = {"v2": kyotolmv2, "v4": kyotolmv4}
