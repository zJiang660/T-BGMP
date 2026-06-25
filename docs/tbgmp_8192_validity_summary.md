# T-BGMP 8192 validity summary

## Fair compression evidence
- qwen25/math uniform_k4v2_rw128: status=failed, found=False, saving=79.223
- qwen25/math uniform_k6v2_rw128: status=success, found=True, saving=66.917
- qwen25/math uniform_k6v4_rw128: status=success, found=True, saving=60.764
- qwen25/math qwen25_tbgmp_minimal: status=failed, found=False, saving=78.198
- qwen25/math qwen25_tbgmp_top6_keys: status=success, found=True, saving=77.172
- qwen25/literature uniform_k4v2_rw128: status=failed, found=False, saving=79.223
- qwen25/literature uniform_k6v2_rw128: status=success, found=True, saving=66.917
- qwen25/literature uniform_k6v4_rw128: status=success, found=True, saving=60.763
- qwen25/literature qwen25_tbgmp_minimal: status=success, found=True, saving=78.198
- qwen25/literature qwen25_tbgmp_top6_keys: status=success, found=True, saving=77.172
- qwen25/science uniform_k4v2_rw128: status=failed, found=False, saving=79.223
- qwen25/science uniform_k6v2_rw128: status=success, found=True, saving=66.917
- qwen25/science uniform_k6v4_rw128: status=success, found=True, saving=60.764
- qwen25/science qwen25_tbgmp_minimal: status=success, found=True, saving=78.198
- qwen25/science qwen25_tbgmp_top6_keys: status=success, found=True, saving=77.172
- falcon3/literature falcon3_tbgmp_minimal: status=success, found=True, saving=79.610
- falcon3/literature falcon3_tbgmp_top6_keys: status=success, found=True, saving=76.973
- falcon3/science uniform_k4v2_rw128: status=success, found=True, saving=79.610
- falcon3/science uniform_k6v2_rw128: status=success, found=True, saving=67.303
- falcon3/science uniform_k6v4_rw128: status=success, found=True, saving=61.150
- falcon3/science falcon3_tbgmp_minimal: status=success, found=True, saving=79.610
- falcon3/science falcon3_tbgmp_top6_keys: status=success, found=True, saving=76.973

## FP16 references
- qwen25/math fp16_baseline: status=success, found=True, saving=0.000
- qwen25/literature fp16_baseline: status=success, found=True, saving=0.000
- qwen25/science fp16_baseline: status=success, found=True, saving=0.000
- falcon3/math fp16_baseline: status=success, found=True, saving=0.000
- falcon3/literature fp16_baseline: status=success, found=True, saving=0.000
- falcon3/science fp16_baseline: status=success, found=True, saving=0.000

## Model capability limitations
These settings are excluded from fair compression claims because FP16 failed.
- zephyr/literature/8192
- zephyr/math/8192
- zephyr/science/8192

## Clean rerun required
- falcon3/math uniform_k4v2_rw128: status=oom
- falcon3/math uniform_k6v2_rw128: status=oom
- falcon3/math uniform_k6v4_rw128: status=oom
- falcon3/math falcon3_tbgmp_minimal: status=oom
- falcon3/math falcon3_tbgmp_top6_keys: status=oom
- falcon3/literature uniform_k4v2_rw128: status=oom
- falcon3/literature uniform_k6v2_rw128: status=oom
- falcon3/literature uniform_k6v4_rw128: status=oom

## Strong positive Qwen2.5 cases
- qwen25/literature/8192: Uniform K4/V2 failed while T-BGMP top6 succeeded.
- qwen25/math/8192: Uniform K4/V2 failed while T-BGMP top6 succeeded.
- qwen25/science/8192: Uniform K4/V2 failed while T-BGMP top6 succeeded.
