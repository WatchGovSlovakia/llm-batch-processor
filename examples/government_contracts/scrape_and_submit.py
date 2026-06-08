"""
Daily job: scrape new intervention requests and start an llm-batch pipeline for each.
Runs once per day from GitHub Actions. State lives in a separate "data" git repo.

GitHub Actions (daily):
  - git pull (data repo)
  - python scrape_and_submit.py
  - git commit && git push
"""

import datetime
import sys

from llm_batch import Pipeline, FileStorage

DATA_DIR = "./data/state"
DEADLINE_DAYS = 6  # legal signup window (5-10d) minus NGO reading time (~3-4d)


def scrape_new_requests() -> list[dict]:
    """Fetch new intervention requests from the government portal."""
    # ... scraping logic ...
    return []  # [{"id": "abc123", "pdf_url": "...", "deadline": "2026-06-15", ...}]


def already_processed(request_id: str) -> bool:
    """Skip requests we've already started a pipeline for."""
    import os
    return os.path.isdir(f"{DATA_DIR}/{request_id}")


def main():
    requests = scrape_new_requests()
    started = 0

    for req in requests:
        if already_processed(req["id"]):
            continue

        deadline = datetime.datetime.fromisoformat(req["deadline"]) - datetime.timedelta(days=3)

        pipeline = Pipeline(
            state_dir=f"{DATA_DIR}/{req['id']}/",
            mode="batch",
            deadline=deadline.isoformat(),
            max_batch_turnaround_hours=24,
            rerun_cmd=f"python continue_pipeline.py {req['id']}",
            storage=FileStorage(f"{DATA_DIR}/{req['id']}/"),
            cleanup_provider_files=True,
        )
        pipeline.save_meta({"pdf_url": req["pdf_url"], "deadline": req["deadline"]})

        # Kick off the first step immediately (will submit as batch and return None)
        import subprocess
        subprocess.run(["python", "continue_pipeline.py", req["id"]], check=True)
        started += 1

    print(f"Started {started} new pipelines.")


if __name__ == "__main__":
    main()
