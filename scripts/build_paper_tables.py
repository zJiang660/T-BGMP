from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "paper_tables"
TABLE_DIR = ROOT / "tables" / "paper"


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    main_df = pd.read_csv(DATA_DIR / "table_main_evidence.csv")
    controls = pd.read_csv(DATA_DIR / "table_control_statistics.csv")
    supporting = pd.read_csv(DATA_DIR / "table_supporting_models.csv")

    restored = main_df["recovered"].str.split("/").str[0].astype(int).sum()
    sensitive = main_df["sensitive"].astype(int).sum()

    print(f"Main evidence models: {len(main_df)}")
    print(f"Conditional restoration: {restored}/{sensitive}")
    print(f"Control-statistics rows: {len(controls)}")
    print(f"Supporting/boundary-supporting rows: {len(supporting)}")
    print("Available paper-ready tables:")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        output = TABLE_DIR / f"{path.stem}.md"
        output.write_text(markdown_table(df), encoding="utf-8")
        print(f"  - {path.name}: {len(df)} rows -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
