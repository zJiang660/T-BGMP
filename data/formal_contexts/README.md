# Formal Retrieval Contexts

These four bounded text snapshots are the domain sources used by the original
single-needle HPC experiment. They are included so that case generation does
not depend on mutable web APIs or dataset streaming order.

`manifest.yaml` records the upstream source description, repository-normalized
size, SHA-256 digest, and complete 72-case grid. The source files use LF line
endings. The formal generator strips blank lines and surrounding whitespace in
the same way as the original HPC runner, then applies the recorded deterministic
seed rotation, insertion jitter, and hidden answers.

The snapshots are inputs, not model outputs. Do not replace them when trying to
reproduce the reported experiment. A new corpus may be supplied for a new
experiment, but it must receive a new protocol name and manifest.
