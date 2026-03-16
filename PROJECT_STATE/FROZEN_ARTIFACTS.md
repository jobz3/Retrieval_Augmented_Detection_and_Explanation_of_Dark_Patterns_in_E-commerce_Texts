## Frozen Artifacts

### Locked data

- `data/processed/ec_darkpattern_phase1_locked/train.csv`
- `data/processed/ec_darkpattern_phase1_locked/val.csv`
- `data/processed/ec_darkpattern_phase1_locked/test.csv`
- `data/processed/ec_darkpattern_phase1_locked/label_map.json`
- `data/processed/ec_darkpattern_phase1_locked/raw_audit.json`
- `data/processed/ec_darkpattern_phase1_locked/split_manifest.json`

### Frozen plain baselines

- `outputs/models/bert/`
- `outputs/models/roberta/`
- `results/baselines/bert.json`

### Frozen weighted baselines

- `outputs/models_weighted/bert_weighted_ce/`
- `outputs/models_weighted/roberta_weighted_ce/`
- `results/baselines_weighted/bert_weighted_ce.json`
- `results/baselines_weighted/roberta_weighted_ce.json`
- `results/analysis/phase1b/`

### Experimental but separate

- `outputs/models_seed_robustness/`
- `results/seed_robustness/`
- `results/analysis/phase1c_seed_robustness/`

### Artifact warning

- `results/baselines/roberta.json` is stale and must not be used as truth for the locked split.
