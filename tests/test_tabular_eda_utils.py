from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "projects" / "educational" / "tabular-eda-utilities" / "tabular_eda_utils.py"
)
STUDENT_META_PATH = ROOT / "datasets" / "student-performance" / "kernel-metadata.json"


def load_module():
    spec = importlib.util.spec_from_file_location("tabular_eda_utils", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tabular_eda_utils_profiles_csv(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "name,score,age,city\nAlice,10,30,\nBob,20,31,Boston\nCara,,29,Austin\n",
        encoding="utf-8",
    )

    mod = load_module()
    rows = mod.load_csv_rows(str(csv_path))

    assert mod.csv_shape(str(csv_path)) == (3, 4)
    assert mod.detect_numeric_columns(rows) == ["score", "age"]

    profile = mod.numeric_profile(rows, ["score", "age"])
    assert profile["score"]["count"] == 2.0
    assert profile["score"]["mean"] == 15.0
    assert profile["age"]["min"] == 29.0
    assert profile["age"]["max"] == 31.0

    missing = mod.missing_rate(rows)
    assert missing["score"] == 1 / 3
    assert missing["city"] == 1 / 3


def test_student_performance_notebook_references_utility_script():
    meta = json.loads(STUDENT_META_PATH.read_text(encoding="utf-8"))

    assert "lorenzoscaturchio/tabular-eda-utilities-for-kaggle-projects" in meta["kernel_sources"]
