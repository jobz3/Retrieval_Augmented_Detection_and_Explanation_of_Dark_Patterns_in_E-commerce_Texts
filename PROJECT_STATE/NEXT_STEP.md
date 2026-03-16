## Next Step

- Phase 1 and Phase 1b are complete enough to move into Phase 2 planning and staged implementation.
- Immediate recommended next step:
  - approve the minimal Phase 2 execution order
  - implement a shared prompt/template layer
  - implement `zero_shot.py` first without retrieval
  - implement `random_few_shot.py` second without retrieval
  - implement `rag_pipeline.py` third as the only pipeline that calls retrieval
- Keep using the locked split as the reference evaluation split unless a stronger experimental reason is documented.
- Continue to report rare-class caveats explicitly, especially for `Forced Action` and `Sneaking`.
