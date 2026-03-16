## Phase 1b Status

- Status: accepted and frozen as the first imbalance-aware ablation
- Intervention scope: class-weighted cross-entropy only
- Locked split used: `data/processed/ec_darkpattern_phase1_locked/`
- Accepted weighted artifacts:
  - BERT weighted results: `results/baselines_weighted/bert_weighted_ce.json`
  - RoBERTa weighted results: `results/baselines_weighted/roberta_weighted_ce.json`
  - BERT weighted checkpoint: `outputs/models_weighted/bert_weighted_ce/`
  - RoBERTa weighted checkpoint: `outputs/models_weighted/roberta_weighted_ce/`
- Accepted Phase 1b comparison summary:
  - `results/analysis/phase1b/overall_comparison.csv`
  - `results/analysis/phase1b/overall_comparison.md`
- Accepted Phase 1b findings:
  - weighted BERT is worse than plain BERT on the locked split
  - weighted RoBERTa is better than plain RoBERTa on the locked split
  - the weighted RoBERTa gain is fragile because `Forced Action` test support is `1` and `Sneaking` test support is `2`
- Current best-performing baseline family:
  - RoBERTa is stronger than BERT on the locked split
  - weighted RoBERTa is the best single-seed Phase 1b result, but it requires robustness caveats
