"""Download AirGapAgent-R (pinned revision) into data/raw/.   Run:  python scripts/download_data.py [--force]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_llm.download import download_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    manifest = download_all(force=ap.parse_args().force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
