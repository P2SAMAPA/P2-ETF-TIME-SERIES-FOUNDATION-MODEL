"""
push_results.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Pushes the daily forecast JSON to the dedicated HuggingFace results dataset.
Uses HfApi.upload_file (NOT HfFileSystem.open, which silently fails on write
per the established suite-wide bug fix).
"""

import json
import os
import tempfile
import datetime as dt

from huggingface_hub import HfApi

from config import OUTPUT_REPO
from trainer import run_daily_forecast


def push_to_hf(payload: dict) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    run_date = payload["run_date"]
    filename_dated = f"results/{run_date}.json"
    filename_latest = "results/latest.json"

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, "payload.json")
        with open(local_path, "w") as f:
            json.dump(payload, f, indent=2)

        for path_in_repo in (filename_dated, filename_latest):
            print(f"[push_results] Uploading to {OUTPUT_REPO}:{path_in_repo}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=OUTPUT_REPO,
                repo_type="dataset",
            )

    print(f"[push_results] Push complete for {run_date}.")


def main():
    payload = run_daily_forecast()
    push_to_hf(payload)


if __name__ == "__main__":
    main()
