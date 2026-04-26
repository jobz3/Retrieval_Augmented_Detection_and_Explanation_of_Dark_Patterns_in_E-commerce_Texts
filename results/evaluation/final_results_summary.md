# Dark Patterns evaluation summary

## High-k sweep
- English best k: 2
- English sweep: {'1': 0.8792, '2': 0.9341, '3': 0.8828, '5': 0.9228, '7': 0.8747, '10': 0.7837, '15': 0.8675}
- German best k: 3
- German sweep: {'1': 0.6991, '2': 0.704, '3': 0.8362, '5': 0.8204, '7': 0.8026, '10': 0.6954, '15': 0.8194}

## Final pipeline artifacts
- English final: results/pipelines/rag_sbert_knn_k2_final.jsonl
- German best artifact: results/pipelines/german_rag_multilingual_de_knn_k3_improved.jsonl

## Rationale judge on k=5
- mean_composite: 18.351
- mean_coherence: 4.902
- mean_faithfulness: 4.782
- mean_specificity: 4.747
- mean_non_circularity: 3.92

## Rewrite quality on k=5
- mean_j_score: 0.8462
- mean_sta: 0.9319
- mean_sim: 0.7605
- mean_fl: None
- entity_leak_rate: 0.1303

## Rationale judge on final English k=2
- mean_composite: 18.237
- mean_coherence: 4.84
- mean_faithfulness: 4.753
- mean_specificity: 4.636
- mean_non_circularity: 4.008

## Rewrite quality on final English k=2
- mean_j_score: 0.6492
- mean_sta: 0.9329
- mean_sim: 0.7631
- mean_fl: 0.2514
- entity_leak_rate: 0.125
