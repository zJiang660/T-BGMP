"""Domain-held-out runner recovered from the formal GPU experiment.

The runtime module and all model/output paths are supplied by command line.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tbgmp.profiling import (  # noqa: E402
    flatten_key_tensor,
    get_cache_layer,
    layer_distortion_metrics,
    load_compressor,
    sample_rows,
)
from tbgmp.risk_score import compute_risk_scores  # noqa: E402
from tbgmp.extension_runtime import create_extension_runtime  # noqa: E402


PROFILE_SCORE_PROTOCOL = "paper_full_normalized_v1"


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The domain-held-out GPU runner requires PyTorch."
        ) from exc
    return torch


def import_hpc_runner(turboquant_root: Path):
    return create_extension_runtime(turboquant_root)


def register_qwen3_4b(hpc, model_path: str) -> None:
    """Add the exact Qwen3-4B entry without editing the shared HPC runner."""
    hpc.MODEL_REGISTRY["qwen3_4b"] = {
        "model_name": "Qwen3-4B-Instruct-2507",
        "default_path": model_path,
        "preferred_dtype": "bfloat16",
        "family": "Qwen3",
        "scale": "4B",
        "prompt_mode": "chat_template",
        "decoding_mode": "deterministic",
        "max_new_tokens": 24,
    }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()


def bool_field(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def done_key(row: dict) -> tuple:
    return (
        row.get("model_id"),
        row.get("domain"),
        str(row.get("context_length")),
        str(row.get("needle_depth")),
        str(row.get("seed")),
        row.get("policy_name"),
    )


def completed_keys(rows: list[dict]) -> set[tuple]:
    return {done_key(row) for row in rows if bool_field(row.get("completed"))}


def finalize_profile_rows(rows: list[dict]) -> list[dict]:
    """Apply the paper's normalized Full score and return rank-ordered rows."""
    ranked = compute_risk_scores(pd.DataFrame(rows))
    ranked["score_protocol"] = PROFILE_SCORE_PROTOCOL
    return ranked.to_dict(orient="records")


def profile_domains(hpc, model, tokenizer, dims, args, model_name: str) -> list[int]:
    torch = require_torch()
    out = Path(args.output_dir)
    risk_csv = out / f"{args.output_prefix}_heldout_risk_ranking.csv"
    used_csv = out / f"{args.output_prefix}_heldout_profile_used_cases.csv"
    if risk_csv.exists() and risk_csv.stat().st_size > 0:
        rows = read_csv(risk_csv)
        protocols = {row.get("score_protocol", "") for row in rows}
        if protocols != {PROFILE_SCORE_PROTOCOL}:
            raise RuntimeError(
                f"Existing held-out ranking {risk_csv} does not use "
                f"{PROFILE_SCORE_PROTOCOL}. Use a clean output directory so "
                "legacy and canonical rankings cannot be mixed."
            )
        ranked = [int(r["layer"]) for r in sorted(rows, key=lambda r: int(r["rank"]))]
        if ranked:
            return ranked

    profile_domains_list = [x.strip() for x in args.profile_domains.split(",") if x.strip()]
    if not profile_domains_list:
        raise ValueError("--profile-domains must contain at least one domain")
    sampled_keys: dict[int, list] = defaultdict(list)
    source_by_layer: dict[int, list[str]] = defaultdict(list)
    used_cases = []
    rows_per_domain = max(
        1,
        (int(args.profile_samples_per_layer) + len(profile_domains_list) - 1)
        // len(profile_domains_list),
    )
    for domain_index, domain in enumerate(profile_domains_list):
        prompt = hpc.build_prompt(
            tokenizer,
            args.model_id,
            domain,
            args.profile_context_length,
            args.profile_depth,
            args.profile_seed,
            hpc.SEEDS[int(args.profile_seed)],
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        actual_tokens = int(input_ids.shape[1])
        used_cases.append(
            {
                "model_id": args.model_id,
                "model_name": model_name,
                "domain": domain,
                "context_length": args.profile_context_length,
                "actual_context_length": actual_tokens,
                "needle_depth": args.profile_depth,
                "seed": args.profile_seed,
                "needle_string": hpc.SEEDS[int(args.profile_seed)],
            }
        )
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        layer_count = int(model.config.num_hidden_layers)
        for layer_idx in range(layer_count):
            keys, _ = get_cache_layer(outputs.past_key_values, layer_idx)
            flat = flatten_key_tensor(keys, torch)
            sampled = sample_rows(
                flat,
                rows_per_domain,
                int(args.profile_seed) + domain_index * 100003 + layer_idx,
                torch,
            )
            sampled_keys[layer_idx].append(sampled)
            source_by_layer[layer_idx].append(
                f"heldout_profile_{domain}_ctx{args.profile_context_length}"
                f"_depth{args.profile_depth}_seed{args.profile_seed}"
            )
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    compressor_class = load_compressor(Path(args.turboquant_root))
    rows = []
    for layer, samples in sorted(sampled_keys.items()):
        combined = torch.cat(samples, dim=0)
        mse_p95, ip_p95, effective_dim = layer_distortion_metrics(
            combined,
            layer=layer,
            bits=int(args.profile_bits),
            max_rows=int(args.profile_samples_per_layer),
            ip_pairs=int(args.profile_ip_pairs),
            seed=int(args.profile_seed),
            compressor_class=compressor_class,
            torch_module=torch,
        )
        rows.append(
            {
                "model_id": args.model_id,
                "model_name": model_name,
                "layer": layer,
                "kv_type": "key",
                "mse_p95": mse_p95,
                "ip_p95": ip_p95,
                "effective_dim": effective_dim,
                "c_mse_upper95": mse_p95,
                "c_ip_upper95": ip_p95,
                "effective_dimension": effective_dim,
                "quant_bits": int(args.profile_bits),
                "profile_samples_per_layer": int(args.profile_samples_per_layer),
                "profile_ip_pairs": int(args.profile_ip_pairs),
                "profile_domains": args.profile_domains,
                "source": ";".join(source_by_layer[layer]),
                "completed": True,
                "error": "",
            }
        )
    rows = finalize_profile_rows(rows)
    fields = list(rows[0].keys())
    write_csv(risk_csv, rows, fields)
    write_csv(
        used_csv,
        used_cases,
        [
            "model_id",
            "model_name",
            "domain",
            "context_length",
            "actual_context_length",
            "needle_depth",
            "seed",
            "needle_string",
        ],
    )
    return [int(r["layer"]) for r in rows]


def load_eval_cases(args) -> list[dict]:
    rows = read_csv(Path(args.case_classification))
    eval_domains = {x.strip() for x in args.eval_domains.split(",") if x.strip()}
    for row in rows:
        case_type = str(row.get("case_type", "")).strip().lower().replace("_", "-")
        if case_type == "k2-sensitive":
            row["case_type"] = "K2-sensitive"
        elif case_type == "k4-sensitive":
            row["case_type"] = "K4-sensitive"
    return [
        row
        for row in rows
        if row.get("domain") in eval_domains
        and row.get("case_type") in {"K4-sensitive", "K2-sensitive"}
    ]


def run_eval(hpc, model, tokenizer, dims, args, model_name: str, ranked_layers: list[int]) -> None:
    out = Path(args.output_dir)
    topk_csv = out / f"{args.output_prefix}_heldout_topk_recovery_science_code.csv"
    control_csv = out / f"{args.output_prefix}_heldout_random_bottom_controls_science_code.csv"
    eval_csv = out / f"{args.output_prefix}_science_code_eval.csv"
    cases_csv = out / f"{args.output_prefix}_heldout_science_code_sensitive_cases.csv"
    eval_cases = load_eval_cases(args)
    write_csv(
        cases_csv,
        eval_cases,
        [
            "model_id",
            "model_name",
            "domain",
            "context_length",
            "needle_depth",
            "seed",
            "needle_string",
            "case_type",
        ],
    )

    topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]
    random_seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
    existing_top = read_csv(topk_csv)
    existing_controls = read_csv(control_csv)
    top_done = completed_keys(existing_top)
    control_done = completed_keys(existing_controls)

    for case in eval_cases:
        domain = case["domain"]
        ctx = int(case["context_length"])
        depth = int(case["needle_depth"])
        seed = int(case["seed"])
        needle_string = case["needle_string"]
        case_type = case["case_type"]
        top_rows_for_case = [
            row
            for row in read_csv(topk_csv)
            if row.get("domain") == domain
            and str(row.get("context_length")) == str(ctx)
            and str(row.get("needle_depth")) == str(depth)
            and str(row.get("seed")) == str(seed)
        ]
        for k in topk_list:
            policy = hpc.tbgmp_policy(args.model_id, k, ranked_layers, case_type)
            policy["policy_name"] = f"{args.model_id}_heldout_top{k}_keys"
            probe = hpc.base_row(
                args.model_id,
                model_name,
                domain,
                ctx,
                depth,
                seed,
                needle_string,
                policy,
                "heldout_topk_recovery",
                case_type,
            )
            if done_key(probe) in top_done:
                continue
            print(
                f"[heldout-topk] {domain} ctx={ctx} depth={depth} seed={seed} {case_type} top{k}",
                flush=True,
            )
            row = hpc.run_generation(
                model,
                tokenizer,
                dims,
                args.model_id,
                model_name,
                domain,
                ctx,
                depth,
                seed,
                needle_string,
                policy,
                "heldout_topk_recovery",
                case_type,
            )
            row["heldout_split"] = "domain_math_literature_to_science_code"
            append_csv(topk_csv, row, hpc.FIELDS + ["heldout_split"])
            top_rows_for_case.append(row)
            top_done.add(done_key(row))

        successful = [
            row
            for row in top_rows_for_case
            if bool_field(row.get("found"))
            and not bool_field(row.get("oom"))
            and not row.get("error")
        ]
        if not successful:
            continue
        def topk_from_policy(row):
            name = str(row.get("policy_name", ""))
            return int(name.split("_top", 1)[1].split("_", 1)[0])
        first = min(successful, key=topk_from_policy)
        first_k = topk_from_policy(first)
        controls = []
        top_same = hpc.tbgmp_policy(args.model_id, first_k, ranked_layers, case_type)
        top_same["policy_name"] = f"{args.model_id}_heldout_samebudget_top{first_k}_keys"
        controls.append(top_same)
        for rs in random_seeds:
            controls.append(hpc.control_policy(args.model_id, "random", first_k, ranked_layers, case_type, seed=rs))
            controls[-1]["policy_name"] = f"{args.model_id}_heldout_random{first_k}_keys_seed{rs}"
        controls.append(hpc.control_policy(args.model_id, "bottom", first_k, ranked_layers, case_type))
        controls[-1]["policy_name"] = f"{args.model_id}_heldout_bottom{first_k}_keys"
        for policy in controls:
            probe = hpc.base_row(
                args.model_id,
                model_name,
                domain,
                ctx,
                depth,
                seed,
                needle_string,
                policy,
                "heldout_same_budget_control",
                case_type,
            )
            if done_key(probe) in control_done:
                continue
            print(
                f"[heldout-control] {domain} ctx={ctx} depth={depth} seed={seed} policy={policy['policy_name']}",
                flush=True,
            )
            row = hpc.run_generation(
                model,
                tokenizer,
                dims,
                args.model_id,
                model_name,
                domain,
                ctx,
                depth,
                seed,
                needle_string,
                policy,
                "heldout_same_budget_control",
                case_type,
            )
            row["heldout_split"] = "domain_math_literature_to_science_code"
            append_csv(control_csv, row, hpc.FIELDS + ["heldout_split"])
            control_done.add(done_key(row))

    all_rows = read_csv(topk_csv) + read_csv(control_csv)
    if all_rows:
        write_csv(eval_csv, all_rows, list(dict.fromkeys([*hpc.FIELDS, "heldout_split"])))


def summarize(args, model_name: str) -> None:
    out = Path(args.output_dir)
    topk = read_csv(out / f"{args.output_prefix}_heldout_topk_recovery_science_code.csv")
    controls = read_csv(out / f"{args.output_prefix}_heldout_random_bottom_controls_science_code.csv")
    eval_cases = read_csv(out / f"{args.output_prefix}_heldout_science_code_sensitive_cases.csv")
    case_keys = {
        (r["domain"], str(r["context_length"]), str(r["needle_depth"]), str(r["seed"]))
        for r in eval_cases
    }
    top_by_case = defaultdict(list)
    for row in topk:
        top_by_case[(row["domain"], str(row["context_length"]), str(row["needle_depth"]), str(row["seed"]))].append(row)
    ctrl_by_case = defaultdict(list)
    for row in controls:
        ctrl_by_case[(row["domain"], str(row["context_length"]), str(row["needle_depth"]), str(row["seed"]))].append(row)

    top_restored = 0
    first_ks = []
    kv_savings = []
    random_restored = 0
    bottom_restored = 0
    for key in case_keys:
        tops = top_by_case.get(key, [])
        successes = [r for r in tops if bool_field(r.get("found")) and not r.get("error")]
        if successes:
            top_restored += 1
            def get_k(row):
                return int(str(row.get("policy_name")).split("_top", 1)[1].split("_", 1)[0])
            first = min(successes, key=get_k)
            first_ks.append(get_k(first))
            if str(first.get("kv_saving_percent", "")).strip():
                kv_savings.append(float(first["kv_saving_percent"]))
        ctrls = ctrl_by_case.get(key, [])
        random_rows = [r for r in ctrls if "random" in str(r.get("policy_name", "")).lower()]
        bottom_rows = [r for r in ctrls if "bottom" in str(r.get("policy_name", "")).lower()]
        if any(bool_field(r.get("found")) and not r.get("error") for r in random_rows):
            random_restored += 1
        if any(bool_field(r.get("found")) and not r.get("error") for r in bottom_rows):
            bottom_restored += 1
    n = len(case_keys)
    row = {
        "Model": model_name,
        "Split type": "domain held-out",
        "Profiling domains": args.profile_domains,
        "Evaluation domains": args.eval_domains,
        "Evaluation sensitive cases": n,
        "Top-k restored": top_restored,
        "Top-k recovery %": top_restored / n if n else "",
        "Random-k restored": random_restored,
        "Random-k recovery %": random_restored / n if n else "",
        "Bottom-k restored": bottom_restored,
        "Bottom-k recovery %": bottom_restored / n if n else "",
        "Median first-success k": sorted(first_ks)[len(first_ks) // 2] if first_ks else "",
        "Max first-success k": max(first_ks) if first_ks else "",
        "Top-k vs Random difference": (top_restored - random_restored) / n if n else "",
        "Top-k vs Bottom difference": (top_restored - bottom_restored) / n if n else "",
        "Fisher p-value if valid": "NA",
        "KV saving if available": sum(kv_savings) / len(kv_savings) if kv_savings else "",
        "Conclusion": (
            "useful" if n >= 10 and top_restored > random_restored and top_restored > bottom_restored
            else "not useful or incomplete"
        ),
    }
    write_csv(out / "heldout_split_summary.csv", [row], list(row.keys()))
    write_csv(out / "qwen3_4b_heldout_summary.csv", [row], list(row.keys()))
    details = topk + controls
    if details:
        write_csv(out / "heldout_split_details.csv", details, list(dict.fromkeys([*details[0].keys()])))
    report = out / "HELDOUT_DOMAIN_SPLIT_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# Held-out Domain Split Sanity Check",
                "",
                "## 1. Executive Summary",
                f"- Completed: {'YES' if n and topk else 'NO'}",
                f"- Model: {model_name}",
                f"- Split: {args.profile_domains} -> {args.eval_domains}",
                f"- Recommended for paper: {'PARTIAL' if row['Conclusion'] == 'useful' else 'NO'}",
                f"- Main reason: {row['Conclusion']}",
                "",
                "## 2. Data and Runtime",
                "- Existing data reused: sensitive case classification from prior Qwen3-4B run.",
                "- New GPU jobs: this held-out wrapper only.",
                "- Model paths kept private: yes.",
                "",
                "## 3. Profiling Setup",
                f"- profiling domains: {args.profile_domains}",
                f"- profile prompts: context={args.profile_context_length}, depth={args.profile_depth}, seed={args.profile_seed}",
                f"- ranking output: `{args.output_prefix}_heldout_risk_ranking.csv`",
                "",
                "## 4. Evaluation Setup",
                f"- evaluation domains: {args.eval_domains}",
                f"- sensitive cases: {n}",
                f"- policies: Top-k {args.topk_list}; Random/Bottom same-budget controls",
                f"- random seeds: {args.random_seeds}",
                "",
                "## 5. Results",
                (
                    "| Model | Eval sensitive | Top-k | Random-k | Bottom-k | Median k | Max k | Difference | p-value | Conclusion |\n"
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
                    f"| {model_name} | {n} | {top_restored} | {random_restored} | {bottom_restored} | "
                    f"{row['Median first-success k']} | {row['Max first-success k']} | "
                    f"{row['Top-k vs Random difference']} | NA | {row['Conclusion']} |"
                ),
                "",
                "## 6. Interpretation",
                "- This is a small domain-held-out sanity check, not a full benchmark.",
                "- Qwen3-4B is evaluated after profiling only math and literature domains.",
                "",
                "## 7. Caveats",
                "- Domain split only.",
                "- Needle retrieval only.",
                "- No LongBench/RULER/reasoning benchmark.",
                "- Diagnostic protocol only.",
                "",
                "## 8. Suggested Paper Use",
                "- Use only if the Top-k separation is clear after manual review.",
                "- Use this only as a held-out split sanity check, not as a broad benchmark.",
                "",
                "## 9. Files Generated",
                f"- `{args.output_prefix}_heldout_risk_ranking.csv`",
                f"- `{args.output_prefix}_heldout_science_code_sensitive_cases.csv`",
                f"- `{args.output_prefix}_heldout_topk_recovery_science_code.csv`",
                f"- `{args.output_prefix}_heldout_random_bottom_controls_science_code.csv`",
                f"- `{args.output_prefix}_science_code_eval.csv`",
                "- `heldout_split_summary.csv`",
                "- `heldout_split_details.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turboquant-root", required=True)
    parser.add_argument("--model-id", choices=("qwen3_4b", "qwen25_3b_instruct"), default="qwen25_3b_instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="qwen25_3b")
    parser.add_argument("--case-classification", required=True)
    parser.add_argument(
        "--context-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "formal_contexts",
    )
    parser.add_argument("--profile-domains", default="math,literature")
    parser.add_argument("--eval-domains", default="science,code")
    parser.add_argument("--profile-context-length", type=int, default=4096)
    parser.add_argument("--profile-depth", type=int, default=50)
    parser.add_argument("--profile-seed", type=int, default=0)
    parser.add_argument("--profile-bits", type=int, default=2)
    parser.add_argument("--profile-samples-per-layer", type=int, default=512)
    parser.add_argument("--profile-ip-pairs", type=int, default=8192)
    parser.add_argument("--topk-list", default="1,2,4,8,12")
    parser.add_argument("--random-seeds", default="0,1,2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hpc = import_hpc_runner(Path(args.turboquant_root))
    if args.model_id == "qwen3_4b":
        register_qwen3_4b(hpc, args.model_path)
    hpc.DOMAIN_FILES = {
        domain: str(args.context_root / f"{domain}.txt")
        for domain in ("math", "literature", "science", "code")
    }
    os.chdir(str(Path(args.turboquant_root)))
    model_name = hpc.MODEL_REGISTRY[args.model_id]["model_name"]
    (out / "run_notes.md").write_text(
        "\n".join(
            [
                "# Held-out Domain Split Run Notes",
                "",
                f"- started_at: {now_iso()}",
                f"- model_id: {args.model_id}",
                f"- profiling_domains: {args.profile_domains}",
                f"- evaluation_domains: {args.eval_domains}",
                "- no LongBench/RULER/reasoning benchmark",
                "- no model weights copied into outputs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not Path(args.model_path).exists():
        raise FileNotFoundError(args.model_path)
    if not Path(args.case_classification).exists():
        raise FileNotFoundError(args.case_classification)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this script inside a GPU Slurm job.")
    tokenizer, model = hpc.load_model(args.model_path, args.model_id)
    dims = hpc.get_dims(model)
    ranked_layers = profile_domains(hpc, model, tokenizer, dims, args, model_name)
    run_eval(hpc, model, tokenizer, dims, args, model_name, ranked_layers)
    summarize(args, model_name)
    with (out / "run_notes.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- finished_at: {now_iso()}\n")
        handle.write("- exit_status: success\n")


if __name__ == "__main__":
    main()
