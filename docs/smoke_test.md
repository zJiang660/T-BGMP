# Backend Smoke Test

The backend smoke test checks whether a local TurboQuant runtime can perform
one real model generation through the T-BGMP backend contract.

The repository does not include model weights. Configure paths outside version
control:

```bash
export TURBOQUANT_ROOT=/path/to/turboquant-pytorch
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/outputs
```

PowerShell:

```powershell
$env:TURBOQUANT_ROOT="/path/to/turboquant-pytorch"
$env:MODEL_ROOT="/path/to/models"
$env:OUTPUT_ROOT="/path/to/outputs"
```

Run:

```bash
python experiments/smoke_test_backend.py \
  --backend turboquant \
  --turboquant-root "$TURBOQUANT_ROOT" \
  --model-root "$MODEL_ROOT" \
  --model-key qwen25_3b \
  --prompt "The hidden answer is ABC123. What is the hidden answer?" \
  --answer ABC123 \
  --policy fp16 \
  --output "$OUTPUT_ROOT/smoke_test.jsonl"
```

If the backend is not yet bound to arbitrary protected key-layer IDs, this
script fails before writing raw output and points to
`docs/backend_integration.md` and `docs/turboquant_api_findings.md`.

If a real backend is available, convert the raw output:

```bash
python scripts/convert_raw_outputs_to_case_csv.py \
  --input "$OUTPUT_ROOT/smoke_test.jsonl" \
  --output "$OUTPUT_ROOT/smoke_test_case.csv"
```

Do not commit real raw model responses unless they are tiny, sanitized, and
intended as a public fixture.
