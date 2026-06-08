"""
Pipeline continuation script: advances one document through its processing steps.

Called in two ways:
  1. By scrape_and_submit.py when a new document is first discovered.
  2. By the hourly poller (llm-batch poll) via rerun_cmd when a batch job completes.

Idempotent — safe to run repeatedly. Each execute() call either:
  - Returns a result immediately (already completed or just finished).
  - Returns None (still pending) → script exits with code 1 so the poller retries later.

GitHub Actions (hourly):
  - git pull (data repo)
  - llm-batch poll ./data/state   (polls all SENT jobs; calls rerun_cmd for completed ones)
  - git commit && git push
"""

import sys
import fitz  # PyMuPDF

from llm_batch import Pipeline, FileStorage

DATA_DIR = "./data/state"
MODEL   = "vertex_ai/gemini-2.5-flash"


def build_ocr_messages(pdf_path: str, page: int) -> list[dict]:
    """Prompt asking the model to extract text from a PDF page image."""
    # ... encode page as base64 image, build messages ...
    return [{"role": "user", "content": f"Extract all text from page {page} of this document."}]


def build_analysis_messages(full_text: str) -> list[dict]:
    """Prompt asking the model to analyse the intervention request."""
    return [{
        "role": "user",
        "content": (
            "You are an environmental lawyer. Analyse this Slovak government intervention "
            "request for a protected area. Extract: applicant, location, requested activity, "
            "ecological risks, relevant legal provisions, and recommended NGO response.\n\n"
            f"{full_text}"
        ),
    }]


def extract_text_with_fitz(pdf_path: str) -> str | None:
    """Try direct text extraction; return None if the PDF is image-only."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text.strip() or None


def process_document(request_id: str) -> bool:
    """
    Returns True when all steps are done, False when still pending.
    """
    meta = ...  # load from pipeline meta file
    pdf_path = meta["pdf_url"]  # already downloaded
    pdf = fitz.open(pdf_path)
    n_pages = len(pdf)

    pipeline = Pipeline(
        state_dir=f"{DATA_DIR}/{request_id}/",
        mode="batch",
        deadline=meta["deadline"],
        max_batch_turnaround_hours=24,
        rerun_cmd=f"python continue_pipeline.py {request_id}",
        storage=FileStorage(f"{DATA_DIR}/{request_id}/"),
    )

    # ── Step 1: OCR all pages as a single bundle ─────────────────────────────
    #
    # Try fitz first — no LLM call needed if text is already embedded.
    direct_text = extract_text_with_fitz(pdf_path)

    ocr_job = pipeline.job(
        "ocr",
        requests=[build_ocr_messages(pdf_path, p) for p in range(n_pages)],
        model=MODEL,
    )
    if direct_text:
        ocr_job.set_completed(direct_text)   # skip the LLM entirely

    pages = ocr_job.execute()   # list[str] | None
    if pages is None:
        return False            # batch still running → rerun_cmd will pick it up

    full_text = "\n\n".join(pages) if isinstance(pages, list) else pages

    # ── Step 2: Structured legal analysis ────────────────────────────────────
    analysis_job = pipeline.job(
        "analysis",
        messages=build_analysis_messages(full_text),
        model=MODEL,
    )

    analysis = analysis_job.execute()   # str | None
    if analysis is None:
        return False

    # ── All steps done ────────────────────────────────────────────────────────
    print(f"[{request_id}] Done.")
    print(f"  Cost: {pipeline.cost_summary()}")
    save_result(request_id, analysis)
    pipeline.finish()
    return True


def save_result(request_id: str, analysis: str):
    """Write the final analysis to the data repo for the NGO to read."""
    with open(f"./data/results/{request_id}.md", "w") as f:
        f.write(analysis)


def main():
    if len(sys.argv) != 2:
        print("Usage: python continue_pipeline.py <request_id>")
        sys.exit(2)

    request_id = sys.argv[1]
    done = process_document(request_id)
    sys.exit(0 if done else 1)   # 0=done, 1=pending (poller retries), 2=failed (via exception)


if __name__ == "__main__":
    main()
