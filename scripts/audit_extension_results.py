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


def main() -> None:
    audit_heldout()
    audit_frozen()
    audit_ruler()
    audit_weight_sensitivity()
    print("Extension result audit: PASS")


if __name__ == "__main__":
    main()
