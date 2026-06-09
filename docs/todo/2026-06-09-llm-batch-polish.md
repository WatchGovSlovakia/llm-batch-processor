# TODO: llm-batch post-MVP polish

Deferred items from the final code review of the MVP (branch `feat/llm-batch-lib`).
None are correctness-critical for the MVP; the two real behavioral bugs found in
review (save_meta deadline propagation; bundle completeness verification) were
fixed before merge in commit 09db821.

## Operational / robustness

1. **Exponential backoff is recorded but not applied (spec §8.1).**
   `poll_attempts` is incremented and persisted on every transient/pending poll,
   but `Poller.poll_all` polls every SENT job on every sweep regardless. In the
   MVP the effective backoff is just the cron interval (e.g. hourly). To honor the
   spec, skip a job whose `poll_attempts` and `last_polled_at` indicate it is still
   inside an exponential delay window (requires persisting `last_polled_at`).

2. **A non-transient, non-`BatchJobFailed` exception in `poll_batch` aborts the
   whole poller sweep.** `Poller.poll_all` only catches `BatchJobFailed`. A
   malformed provider response (e.g. `KeyError`/`ValueError` while parsing output
   JSONL) would stop the sweep and leave other SENT jobs unprocessed. Consider
   isolating each job: catch `Exception` per job, record it (return dict / log),
   and continue. Weigh against masking real bugs silently.

3. **`create_file` succeeds but `create_batch` fails terminally → orphaned input
   file.** `providers.submit_batch` never returns a handle in this case, so the
   uploaded `input_file_id` is only recoverable via `sweep_orphans` (spec §7.3,
   the documented mitigation). Add a clarifying comment at the `create_batch`
   call site noting this orphan window.

## Consistency / cosmetics

4. **`PostgresStorage.for_job()` does not auto-load, but `FileStorage.for_job()`
   does.** Both are functionally correct because `Job.__init__` always calls
   `load()` explicitly, but the asymmetry is a footgun for direct storage-API
   users. Pick one convention for both backends.

5. **`set_completed()` sets `submitted_at` to the call time** (via `_base_state`),
   which is misleading for a preset job that never submitted anything. Consider
   `submitted_at=None` on the preset path. No functional impact (the poller only
   scans SENT jobs).

## Source

Final review of the MVP implementation. Spec:
`docs/superpowers/specs/2026-06-07-llm-interactive-batch-lib-design.md`.
Plan: `docs/superpowers/plans/2026-06-08-llm-interactive-batch-lib-plan.md`.
