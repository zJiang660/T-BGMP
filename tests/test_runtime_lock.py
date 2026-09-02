import hashlib
from pathlib import Path

from scripts.check_runtime_lock import check_model, check_python


def test_check_python_requires_exact_patch_version():
    actual = check_python("0.0.0")
    assert not actual["passed"]


def test_check_model_validates_content_hash(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    expected = hashlib.sha256(config.read_bytes()).hexdigest()
    checks = check_model(
        tmp_path,
        "example",
        {"local_dir": "model", "config_sha256": expected},
    )
    assert checks == [
        {
            "check": "model:example:config.json",
            "expected": expected,
            "actual": expected,
            "passed": True,
        }
    ]
