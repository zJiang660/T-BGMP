# Claims to Artifacts Mapping

| Paper claim | Evidence files | Verification command | Expected result |
|---|---|---|---|
| Main models restore 183/183 conditional sensitive cases | `results/paper_tables/table_main_evidence.csv`; `results/main_evidence/*`; `scripts/audit_results.py` | `python scripts/audit_results.py` | `Total main: 183/183` |
| Qwen2.5 scale comparison | `results/paper_tables/table_qwen25_scale.csv`; `tables/paper/table_qwen25_scale.md` | `python scripts/build_paper_tables.py` | Qwen2.5-3B and Qwen2.5-14B comparison table regenerated |
| Same-budget controls show Top-k risk guidance is stronger than random/bottom controls | `results/paper_tables/table_control_statistics.csv`; `scripts/audit_results.py` | `python scripts/audit_results.py` | Top-k rows remain 100% for main models while random/bottom controls are lower |
| Gemma2 is a value-bottleneck boundary-supporting case, not main evidence | `results/paper_tables/table_gemma2_boundary.csv`; `results/supporting/gemma2_9b/*`; `scripts/audit_results.py` | `python scripts/audit_results.py` | `K6/V2 7/72`, `K6/V4 72/72` |
| Paper figures are reproducible from repository data | `scripts/build_figures.py`; `figures/paper/` | `python scripts/build_figures.py` | Reproduced figure files under `figures/paper/` |
| Paper tables are reproducible from cleaned CSV | `scripts/build_paper_tables.py`; `results/paper_tables/`; `tables/paper/` | `python scripts/build_paper_tables.py` | Markdown and CSV paper tables regenerated |
| TurboQuant backend smoke path is validated on a small example | `docs/smoke_test.md`; `examples/smoke_test/`; `slurm/xec/submit_smoke_test_a800_template.sbatch` | See `docs/smoke_test.md` | Sanitized smoke raw JSONL and case-level CSV examples are present |
| Artifact integrity and public-safety checks pass | `scripts/check_artifact_integrity.py` | `python scripts/check_artifact_integrity.py` | `ARTIFACT INTEGRITY: PASS` |

The main 183/183 aggregate excludes supporting and boundary/excluded models.
