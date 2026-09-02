from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "extensions"
PAPER = ROOT / "results" / "paper_tables"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def audit_heldout() -> None:
    cases = pd.read_csv(RESULTS / "domain_heldout" / "case_level.csv")
    paper = pd.read_csv(PAPER / "table_domain_heldout.csv").set_index("model")
    require(len(cases) == 72 and cases["completed"].astype(bool).all(), "held-out case set is complete")
    for model, rows in cases.groupby("model"):
        top = int(rows["top_recovered"].sum())
        random_union = int(rows[["random0", "random1", "random2"]].astype(bool).any(axis=1).sum())
        bottom = int(rows["bottom"].sum())
        require(paper.loc[model, "topk"] == f"{top}/{len(rows)}", f"held-out Top-k matches {model}")
        require(paper.loc[model, "randomk"] == f"{random_union}/{len(rows)}", f"held-out Random-k matches {model}")
        require(paper.loc[model, "bottomk"] == f"{bottom}/{len(rows)}", f"held-out Bottom-k matches {model}")


def audit_frozen() -> None:
    cases = pd.read_csv(RESULTS / "frozen_top3" / "case_level.csv")
    paper = pd.read_csv(PAPER / "table_frozen_top3.csv").set_index("model")
    require(len(cases) == 72, "frozen Top3 case set is complete")
    for model, rows in cases.groupby("model"):
        values = {
            "fp16": f"{int(rows['fp16'].sum())}/{len(rows)}",
            "aggressive": f"{int(rows['aggressive'].sum())}/{len(rows)}",
            "uniform": f"{int(rows['uniform'].sum())}/{len(rows)}",
            "frozen_top3": f"{int(rows['frozen_top'].sum())}/{len(rows)}",
            "bottom3": f"{int(rows['frozen_bottom'].sum())}/{len(rows)}",
        }
        random_mean = rows[["random0", "random1", "random2"]].sum().mean()
        values["random3"] = f"{random_mean:.1f}/{len(rows)}".replace(".0/", "/")
        conditional = rows[rows["conditional_failure"].astype(bool)]
        values["recovery"] = f"{int(conditional['recovered'].sum())}/{len(conditional)}"
        for field, expected in values.items():
            require(str(paper.loc[model, field]) == expected, f"frozen {field} matches {model}")


def audit_ruler() -> None:
    screening = pd.read_csv(RESULTS / "ruler" / "screening_case_level.csv")
    recovery = pd.read_csv(RESULTS / "ruler" / "recovery_case_level.csv")
    paper = pd.read_csv(PAPER / "table_ruler_transfer.csv").set_index("model")
    labels = {"qwen3_4b": "Qwen3-4B", "qwen25_3b": "Qwen2.5-3B"}
    require(len(screening) == 600, "RULER screening set is complete")
    require(len(recovery) == 304, "RULER conditional recovery set is complete")
    valid_columns = [column for column in recovery.columns if column.endswith("_valid")]
    require(recovery[valid_columns].astype(bool).all().all(), "all RULER recovery executions are valid")
    for model_id, rows in recovery.groupby("model"):
        model = labels[model_id]
        screen = screening[screening["model"] == model_id]
        require(len(rows) == int(paper.loc[model, "n"]), f"RULER conditional N matches {model}")
        for k in (1, 4, 8, 12):
            observed = round(100.0 * rows[f"top{k}"].mean(), 1)
            require(math.isclose(observed, float(paper.loc[model, f"top{k}_percent"])), f"RULER Top{k} matches {model}")
        cumulative = rows[[f"top{k}" for k in range(1, 13)]].astype(bool).any(axis=1)
        text = f"{int(cumulative.sum())}/{len(rows)} ({100.0 * cumulative.mean():.1f}%)"
        require(text == paper.loc[model, "cumulative_top12"], f"RULER cumulative Top12 matches {model}")
        random_mean = sum(rows[f"random{seed}_k12"].sum() for seed in range(3)) / 3.0
        require(math.isclose(round(100.0 * random_mean / len(rows), 1), float(paper.loc[model, "random12_percent"])), f"RULER Random12 matches {model}")
        require(math.isclose(round(100.0 * rows["bottom12"].mean(), 1), float(paper.loc[model, "bottom12_percent"])), f"RULER Bottom12 matches {model}")
        require(screen[["fp16_valid", "aggressive_valid"]].astype(bool).all().all(), f"RULER screening executions are valid for {model}")


def spearman_from_ranks(left: pd.Series, right: pd.Series) -> float:
    n = len(left)
    squared = ((left.astype(float) - right.astype(float)) ** 2).sum()
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


def audit_weight_sensitivity() -> None:
    rankings = pd.read_csv(RESULTS / "weight_sensitivity" / "rankings.csv")
    cases = pd.read_csv(RESULTS / "weight_sensitivity" / "case_level.csv")
    paper = pd.read_csv(PAPER / "table_weight_sensitivity.csv")
    def weight_token(value: float) -> str:
        return "1" if float(value) == 1.0 else "0p5"

    label = lambda row: "_".join(
        weight_token(value) for value in (row.alpha, row.beta, row.gamma)
    )
    paper = paper.assign(weights=paper.apply(label, axis=1))
    require(len(rankings) == 288, "weight sensitivity contains four complete rankings per model")
    require(len(cases) == 576, "weight sensitivity contains all first-success cases")
    for _, expected in paper.iterrows():
        model = expected["model"]
        weights = expected["weights"]
        base = rankings[(rankings["model"] == model) & (rankings["weights"] == "1_1_1")].set_index("layer")
        current = rankings[(rankings["model"] == model) & (rankings["weights"] == weights)].set_index("layer")
        rho = spearman_from_ranks(base["rank"], current["rank"])
        top3 = len(set(base.nsmallest(3, "rank").index) & set(current.nsmallest(3, "rank").index))
        selected = cases[(cases["model"] == model) & (cases["weights"] == weights)]
        require(math.isclose(round(rho, 4), float(expected["spearman_rho"])), f"weight rank correlation matches {model}/{weights}")
        require(f"{top3}/3" == expected["top3_overlap"], f"weight Top3 overlap matches {model}/{weights}")
        require(math.isclose(round(selected["first_success_k"].mean(), 3), float(expected["mean_first_success_k"])), f"weight first-success mean matches {model}/{weights}")


def audit_fixed_layer() -> None:
    cases = pd.read_csv(RESULTS / "fixed_layer" / "case_level.csv")
    require(len(cases) == 388, "fixed-layer evidence contains all paper-used rows")
    require(cases["completed"].astype(bool).all(), "fixed-layer executions are complete")
    require(set(cases["status"]) <= {"ok", "completed"}, "fixed-layer executions are valid")
    required_variants = {"fixed_l0", "fixed_l0_l7_l25", "tbgmp_top1", "tbgmp_top3"}
    for model, rows in cases.groupby("model"):
        require(set(rows["variant"]) == required_variants, f"fixed-layer variants are complete for {model}")
        counts = rows.groupby("variant")["found"].agg(["sum", "count"])
        print(f"OBSERVED fixed-layer {model}: " + ", ".join(
            f"{variant}={int(values['sum'])}/{int(values['count'])}"
            for variant, values in counts.iterrows()
        ))
    top_rows = cases[cases["variant"].str.startswith("tbgmp_")]
    require(
        set(top_rows["evidence_source"]) == {"canonical_main_topk"}
        and set(top_rows["metric_semantics"]) == {"exact_at_budget"},
        "fixed-layer T-BGMP columns use canonical exact-at-budget evidence",
    )


def audit_cross_seed() -> None:
    cases = pd.read_csv(RESULTS / "cross_seed_heldout" / "case_level.csv")
    rankings = pd.read_csv(RESULTS / "cross_seed_heldout" / "rankings.csv")
    require(len(cases) == 1728, "cross-seed evidence contains all Top1--Top12 rows")
    require(cases["completed"].astype(bool).all(), "cross-seed executions are complete")
    require(set(cases["status"]) == {"ok"}, "cross-seed executions are valid")
    require(len(rankings) == 144, "cross-seed calibration rankings are complete")
    for keys, rows in cases.groupby(["model", "calibration_seed", "evaluation_seed"]):
        require(rows["case_id"].nunique() == 36, f"cross-seed direction has 36 cases for {keys}")
        require(set(rows["k"]) == set(range(1, 13)), f"cross-seed direction has Top1--Top12 for {keys}")
        recovered = rows[rows["found"].astype(bool)].groupby("case_id")["k"].min()
        require(len(recovered) == 36 and int(recovered.max()) <= 8, f"cross-seed cumulative recovery reaches 36/36 by Top8 for {keys}")
        print(
            f"OBSERVED cross-seed {keys}: Top1={int((recovered <= 1).sum())}/36, "
            f"Top4={int((recovered <= 4).sum())}/36, Top8={int((recovered <= 8).sum())}/36"
        )


def audit_gemma_boundary() -> None:
    cases = pd.read_csv(RESULTS / "gemma2_boundary" / "case_level.csv")
    summary = pd.read_csv(RESULTS / "gemma2_boundary" / "summary.csv")
    require(len(cases) == 732, "Gemma2 boundary evidence contains all completed rows")
    require(cases["completed"].astype(bool).all(), "Gemma2 boundary executions are complete")
    require(set(cases["status"]) == {"ok"}, "Gemma2 boundary executions are valid")

    discovery = cases[cases["stage"] == "discovery"]
    for policy, rows in discovery.groupby("policy", sort=False):
        expected = summary[(summary["record_type"] == "baseline") & (summary["policy"] == policy)].iloc[0]
        require(int(rows["found"].sum()) == int(expected["exact_found"]), f"Gemma2 baseline summary matches {policy}")

    recovery = cases[cases["stage"] == "tbgmp_recovery"].copy()
    recovery["k"] = recovery["policy"].str.extract(r"top(\d+)_", expand=False).astype(int)
    recovered: set[str] = set()
    for k in range(1, 13):
        rows = recovery[recovery["k"] == k]
        recovered.update(rows.loc[rows["found"].astype(bool), "case_id"].astype(str))
        expected = summary[(summary["record_type"] == "key_only_topk") & (summary["k"] == k)].iloc[0]
        require(int(rows["found"].sum()) == int(expected["exact_found"]), f"Gemma2 exact Top{k} summary matches case rows")
        require(len(recovered) == int(expected["cumulative_unique_found"]), f"Gemma2 cumulative Top{k} summary matches case union")
    print(f"OBSERVED Gemma2 cumulative Top12={len(recovered)}/25; unrecovered={25-len(recovered)}/25")


def main() -> None:
    audit_heldout()
    audit_frozen()
    audit_ruler()
    audit_weight_sensitivity()
    audit_fixed_layer()
    audit_cross_seed()
    audit_gemma_boundary()
    print("Extension result audit: PASS")


if __name__ == "__main__":
    main()
