from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import parse_bool


DEFAULT_KEY_FIELDS = ("model", "case_id", "stage", "policy")


def _clean_scalar(value: Any) -> Any:
    if not isinstance(value, (list, dict, tuple)):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


class IncrementalResultStore:
    """Durable JSONL attempt journal plus an atomic latest-row CSV snapshot."""

    def __init__(
        self,
        csv_path: Path,
        *,
        key_fields: Iterable[str] = DEFAULT_KEY_FIELDS,
        checkpoint_every: int = 1,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.journal_path = self.csv_path.with_suffix(".jsonl")
        self.key_fields = tuple(key_fields)
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._latest: dict[tuple[str, ...], dict[str, Any]] = {}
        self._completed: set[tuple[str, ...]] = set()
        self._since_snapshot = 0
        self.appended = 0
        self.skipped = 0
        self._load_existing()

    def _key(self, row: dict[str, Any]) -> tuple[str, ...]:
        missing = [field for field in self.key_fields if field not in row]
        if missing:
            raise ValueError(f"result row missing identity fields: {missing}")
        return tuple(str(row.get(field, "")) for field in self.key_fields)

    @staticmethod
    def _is_completed(row: dict[str, Any]) -> bool:
        value = row.get("completed", "")
        if value not in ("", None):
            try:
                return parse_bool(value)
            except ValueError:
                return False
        return str(row.get("status", "")).lower() == "success"

    def _accept(self, row: dict[str, Any]) -> None:
        cleaned = {key: _clean_scalar(value) for key, value in row.items()}
        key = self._key(cleaned)
        self._latest[key] = cleaned
        if self._is_completed(cleaned):
            self._completed.add(key)
        else:
            self._completed.discard(key)

    def _load_existing(self) -> None:
        if self.csv_path.is_file() and self.csv_path.stat().st_size:
            for row in pd.read_csv(self.csv_path).to_dict(orient="records"):
                self._accept(row)
        if self.journal_path.is_file() and self.journal_path.stat().st_size:
            with self.journal_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
            for line_number, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A killed process can leave only its final journal line
                    # incomplete. Earlier durable attempts remain usable.
                    if line_number == len(lines):
                        break
                    raise
                self._accept(row)

    def get(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        return self._latest.get(self._key(identity))

    def completed(self, identity: dict[str, Any]) -> bool:
        key = self._key(identity)
        if key in self._completed:
            self.skipped += 1
            return True
        return False

    def append(self, row: dict[str, Any]) -> None:
        cleaned = {key: _clean_scalar(value) for key, value in row.items()}
        payload = (
            json.dumps(cleaned, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.journal_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._accept(cleaned)
        self.appended += 1
        self._since_snapshot += 1
        if self._since_snapshot >= self.checkpoint_every:
            self.snapshot()

    def rows(self, *, stage: str | None = None) -> list[dict[str, Any]]:
        rows = list(self._latest.values())
        if stage is not None:
            rows = [row for row in rows if str(row.get("stage")) == stage]
        return rows

    def snapshot(self) -> None:
        rows = self.rows()
        temp_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(temp_path, index=False)
        os.replace(temp_path, self.csv_path)
        self._since_snapshot = 0

    def close(self) -> None:
        self.snapshot()
