from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


WEIGHTS = {
    "1_1_1": (1.0, 1.0, 1.0),
    "1_1_0p5": (1.0, 1.0, 0.5),
    "1_0p5_1": (1.0, 0.5, 1.0),
    "0p5_1_1": (0.5, 1.0, 1.0),
}


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / span


def normalized_terms(profile: pd.DataFrame, model: str) -> pd.DataFrame:
    out = pd.DataFrame({"layer": profile["layer"].astype(int)})
    if model == "Qwen3-4B":
        mse = profile["c_mse_upper95"].astype(float)
        ip = profile["c_ip_upper95"].astype(float)
        effective = profile["effective_dimension"].astype(float)
    else:
        mse = profile["mse_proxy"].astype(float)
        ip = profile["ip_distortion_proxy"].astype(float)
        effective = profile["effective_dimension"].astype(float)
    out["mse_norm"] = minmax(mse)
    out["log_ip_norm"] = minmax(ip.map(math.log1p))
    out["inverse_effdim_norm"] = minmax(1.0 / effective)
    return out


def build_rankings(profile: pd.DataFrame, model: str) -> pd.DataFrame:
    terms = normalized_terms(profile, model)
    frames = []
    for label, (alpha, beta, gamma) in WEIGHTS.items():
        ranked = terms.copy()
        ranked["risk_score"] = (
            alpha * ranked["mse_norm"]
            + beta * ranked["log_ip_norm"]
            + gamma * ranked["inverse_effdim_norm"]
        )
        ranked = ranked.sort_values(["risk_score", "layer"], ascending=[False, True])
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["model"] = model
        ranked["weights"] = label
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def case_key(row: pd.Series) -> str:
    return f"{row.domain}_ctx{row.context_length}_d{row.needle_depth}_s{row.seed}"


def qwen3_first_success(details: pd.DataFrame) -> pd.DataFrame:
    variant = {
        "1_1_1": "full_1_1_1",
        "1_1_0p5": "full_1_1_0p5",
        "1_0p5_1": "full_1_0p5_1",
        "0p5_1_1": "full_0p5_1_1",
    }
    rows = []
    group_columns = ["domain", "context_length", "needle_depth", "seed"]
    for label, source_variant in variant.items():
        selected = details[details["variant"] == source_variant]
        for _, group in selected.groupby(group_columns, sort=True):
            successful = group[group["found"].astype(str).str.lower() == "true"]
            first = int(successful["k"].astype(int).min()) if len(successful) else None
            rows.append({
                "model": "Qwen3-4B",
                "case_id": case_key(group.iloc[0]),
                "weights": label,
                "first_success_k": first,
                "recovered_within_top12": int(first is not None),
                "evidence": "direct_gpu_output",
            })
    return pd.DataFrame(rows)


def qwen25_first_success(full_cases: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    base = rankings[rankings["weights"] == "1_1_1"].nsmallest(3, "rank")["layer"].tolist()
    if full_cases["first_success_k"].astype(int).max() > 3:
        raise ValueError("Qwen2.5 output reuse requires every case to recover within Top3")
    rows = []
    for label in WEIGHTS:
        current = rankings[rankings["weights"] == label].nsmallest(3, "rank")["layer"].tolist()
        if current != base:
            raise ValueError(f"Qwen2.5 Top3 prefix changed for {label}")
        for _, source in full_cases.iterrows():
            rows.append({
                "model": "Qwen2.5-3B",
                "case_id": case_key(source),
                "weights": label,
                "first_success_k": int(source["first_success_k"]),
                "recovered_within_top12": int(bool(source["recovered_within_top12"])),
                "evidence": "direct_gpu_output" if label == "1_1_1" else "reused_identical_top3_prefix",
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the offline weight-sensitivity evidence from completed runs.")
    parser.add_argument("--qwen3-profile", type=Path, required=True)
    parser.add_argument("--qwen25-profile", type=Path, required=True)
    parser.add_argument("--qwen3-details", type=Path, required=True)
    parser.add_argument("--qwen25-full-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen3_rankings = build_rankings(pd.read_csv(args.qwen3_profile), "Qwen3-4B")
    qwen25_rankings = build_rankings(pd.read_csv(args.qwen25_profile), "Qwen2.5-3B")
    rankings = pd.concat([qwen3_rankings, qwen25_rankings], ignore_index=True)
    qwen3_cases = qwen3_first_success(pd.read_csv(args.qwen3_details))
    qwen25_source = pd.read_csv(args.qwen25_full_cases)
    if "risk_score" in qwen25_source:
        qwen25_source = qwen25_source[qwen25_source["risk_score"] == "Full"]
    qwen25_cases = qwen25_first_success(qwen25_source, qwen25_rankings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output_dir / "rankings.csv", index=False)
    pd.concat([qwen3_cases, qwen25_cases], ignore_index=True).to_csv(
        args.output_dir / "case_level.csv", index=False
    )


if __name__ == "__main__":
    main()
