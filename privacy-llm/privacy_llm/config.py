"""Single source of truth for dataset identity and on-disk locations."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

HF_REPO_ID = "parameterlab/leaky_thoughts"
# Pinned commit so every analysis runs on byte-identical files.
HF_REVISION = "f1d8f27376eb0854ec5b476e88b942d0060eee62"

# split name -> file name in the HF repo
SPLITS = {
    "full": "airgapagent-r.json",
    "small": "airgapagent-r-small.json",
    "swap_original": "airgapagent-r-ablation-swap-original.json",
    "swap_flipped": "airgapagent-r-ablation-swap-flipped.json",
}


def hf_url(filename: str, revision: str = HF_REVISION) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{revision}/{filename}"
