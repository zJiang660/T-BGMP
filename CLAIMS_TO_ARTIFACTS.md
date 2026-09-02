# Claims to Artifacts Mapping

| Camera-ready item or claim | Canonical evidence | Verification |
|---|---|---|
| Table 1: same-budget Top-k recovery on the four core models | `results/paper_tables/table_control_statistics.csv`; sanitized model bundles under `results/main_evidence/` | `python scripts/audit_results.py` |
| Table 2: first-success Top-k budgets | `results/paper_tables/table_first_success_k.csv` | `python scripts/validate_csv_schema.py` |
| Figure 2: per-domain cumulative recovery | `results/paper_tables/figure_2_domain_recovery.csv` | `python scripts/build_figures.py` |
| Table 3: MSE, MSE + IP, and Full risk ablation | `results/paper_tables/table_risk_ablation.csv` | `python scripts/build_paper_tables.py` |
| Table 4: risk-score weight sensitivity | `results/paper_tables/table_weight_sensitivity.csv`; `results/extensions/weight_sensitivity/` | `python scripts/audit_extension_results.py` |
| Table 5: domain-held-out recovery | `results/paper_tables/table_domain_heldout.csv`; `results/extensions/domain_heldout/case_level.csv` | `python scripts/audit_extension_results.py` |
| Table 6: frozen Top3 proactive protection | `results/paper_tables/table_frozen_top3.csv`; `results/extensions/frozen_top3/case_level.csv` | `python scripts/audit_extension_results.py` |
| Table 7: RULER-style task transfer | `results/paper_tables/table_ruler_transfer.csv`; `results/extensions/ruler/` | `python scripts/audit_extension_results.py` |
| Four-model diagnostic recovery totals 183/183 conditional failures | `results/paper_tables/table_main_evidence.csv`; `results/main_evidence/*` | `python scripts/audit_results.py` |
| Gemma2 is boundary evidence rather than a fifth core model | `results/paper_tables/table_gemma2_boundary.csv`; `results/supporting/gemma2_9b/*` | `python scripts/audit_results.py` |
| The small backend integration path is validated | `docs/smoke_test.md`; `examples/smoke_test/` | See `docs/smoke_test.md` |

Generated Markdown tables and Figure 2 are written to ignored directories
`tables/paper/` and `figures/paper/`. They are build products, not canonical
inputs.
