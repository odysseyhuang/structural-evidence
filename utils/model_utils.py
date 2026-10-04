from pathlib import Path


RLCODER_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = RLCODER_ROOT / "models"


def local_model_path(model_dir_name):
    return str(MODELS_ROOT / model_dir_name)


def resolve_model_path(model_path):
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = Path.cwd() / path
    rlcoder_path = RLCODER_ROOT / path

    if path.parts and path.parts[0] == "models":
        return str(rlcoder_path)
    if cwd_path.exists():
        return str(cwd_path)
    if rlcoder_path.exists():
        return str(rlcoder_path)
    return str(cwd_path)
