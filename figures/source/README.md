# Figure Sources

Paper figures in `../paper/` are generated from the cleaned CSV files in
`results/paper_tables/`. Run:

```bash
python scripts/build_figures.py
```

The minimal repository includes final paper PDFs and a reproducible PNG control
plot. It does not include raw model generations or cluster logs.
