import re
import shutil
import sys
from pathlib import Path

REPO_ID = "facebook/tribev2"
CHECKPOINT_NAME = "best.ckpt"


def check_tribev2():
    try:
        import torch  
        from tribev2.demo_utils import TribeModel  
    except ImportError as e:
        sys.exit(
            f"Missing dependency: {e}\n\n"
            "Install TRIBE v2:\n"
            "  uv pip install --python .venv-tribe "
            '"tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"'
        )


def _fix_config_posixpath(config_path: Path):
    text = config_path.read_text(encoding="utf-8")
    if "pathlib.PosixPath" not in text:
        return

    pattern = re.compile(
        r"!!python/object/apply:pathlib\.PosixPath\s*\n"
        r"((?:\s+-\s+.+\n)+)",
    )

    def _replace(m: re.Match) -> str:
        parts = re.findall(r"-\s+(.+)", m.group(1))
        return "'" + "/".join(p.strip("'\" ") for p in parts) + "'\n"

    config_path.write_text(pattern.sub(_replace, text), encoding="utf-8")


def download_model(cache_dir: str = "./cache") -> Path:
    from huggingface_hub import hf_hub_download

    local_dir = Path(cache_dir) / "tribev2_model"
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("config.yaml", CHECKPOINT_NAME):
        dst = local_dir / filename
        if not dst.exists():
            shutil.copy2(hf_hub_download(REPO_ID, filename), str(dst))

    _fix_config_posixpath(local_dir / "config.yaml")
    return local_dir


def load_model(cache_dir: str = "./cache", device: str = "auto"):
    import torch
    from tribev2.demo_utils import TribeModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return TribeModel.from_pretrained(
        str(download_model(cache_dir)),
        cache_folder=cache_dir,
        device=device,
    )
