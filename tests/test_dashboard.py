"""Dashboard data layer: loads committed results, tolerates live/partial runs."""
import json
import os

import pytest

pd = pytest.importorskip("pandas")

from dashboard import data


def test_benchmark_tables_load_committed_report():
    report = data.load_benchmark()
    assert report is not None
    table = data.benchmark_table(report)
    assert list(table["arm"]) == data.ARM_ORDER
    assert table.set_index("arm").loc["proposed_fl", "Accuracy (%)"] == pytest.approx(84.67, abs=0.01)
    acc = data.acceptance_table(report)
    assert len(acc) == 8 and acc["pass"].sum() == 3


def test_benchmark_rounds_are_long_format():
    df = data.benchmark_rounds()
    assert set(df["arm"]) == set(data.ARM_ORDER)
    assert {"round", "accuracy", "mb_cumulative"} <= set(df.columns)


@pytest.mark.parametrize("study", list(data.ABLATION_STUDIES))
def test_ablation_tables(study):
    assert not data.ablation_table(study).empty


def test_load_rounds_tolerates_partial_last_line(tmp_path):
    with open(tmp_path / "rounds.jsonl", "w") as fh:
        fh.write(json.dumps({"round": 1, "accuracy": 0.2}) + "\n")
        fh.write('{"round": 2, "accur')                       # mid-write
    df = data.load_rounds(str(tmp_path))
    assert list(df["round"]) == [1]


def test_load_rounds_empty_dir(tmp_path):
    assert data.load_rounds(str(tmp_path)).empty


def test_discover_runs_finds_committed_arms():
    names = {r.name for r in data.discover_runs()}
    assert os.path.join("results", "benchmark", "proposed_fl") in names


def test_build_run_command_matches_cli():
    cmd = data.build_run_command("configs/quick.yaml", "x", "runs",
                                 {"train.rounds": 3, "dp.enabled": True})
    assert cmd[1:4] == ["-m", "fedav.cli", "run"]
    assert cmd[-2:] == ["train.rounds=3", "dp.enabled=true"]
