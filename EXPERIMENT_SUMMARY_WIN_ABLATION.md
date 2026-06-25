# Experiment Summary: CPLMP Window Ablation

## Scope
- Task: LecSlides 10% summarization
- Eval size: 11447 samples (full eval)
- Decode: greedy (`num_beams=1`, `max_new_tokens=128`)
- Seed: 42

## Final Metrics (EvalALL)

| Setting | ROUGE-1 | ROUGE-L | METEOR | Notes |
|---|---:|---:|---:|---|
| Baseline (UDOP+T5) | 0.216134 | 0.165096 | 0.100861 | reference |
| W2 | 0.2175288073 | 0.1659772436 | 0.1016839783 | done |
| W3 | 0.2176598733 | 0.1660628255 | 0.1016571877 | done |
| W4 | 0.2174853099 | 0.1659446090 | 0.1016304474 | done |
| W8(Full) | 0.2175282116 | 0.1659013011 | 0.1017167967 | done |

## Key Finding
- ROUGE-best window: **W3**
- METEOR-best window: **W8**
- All tested windows improve over baseline on all three metrics.

## Artifact Paths
- Window summary json:
  - `log/repro/active/analysis/window_ablation/window_ablation_summary.json`
- Per-setting metrics:
  - `log/repro/active/metrics/ablation_WIN2_evalALL.json`
  - `log/repro/active/metrics/ablation_WIN3_evalALL.json`
  - `log/repro/active/metrics/ablation_WIN4_evalALL.json`
  - `log/repro/active/metrics/cplmp_train800_v2_alpha006_evalALL.json`
- Thesis draft updated:
  - `thesis_draft.tex`
