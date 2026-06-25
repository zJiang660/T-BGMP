# Policy Configs

This directory contains small JSON policy files copied from the local
`turboquant-pytorch` source tree after safety checks.

Policies use uniform aggressive defaults plus optional key-layer overrides.
The files are intended for reproducibility review and analysis, not for
shipping model weights or cache artifacts.

Generic YAML configs define the public runner contract:

- `default_experiment.yaml`
- `policies.yaml`
- `prompt_template.yaml`
- `main_models.yaml`
- `supporting_models.yaml`
- `paths_template.yaml`

Machine-specific copies of `paths_template.yaml` should remain outside version
control.
