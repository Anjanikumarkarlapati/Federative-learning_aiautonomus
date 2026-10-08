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


# ------------------------------------------------------------ per-vehicle telemetry

@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from fedav.config import default_config
    from fedav.sim.runner import run_experiment

    out = tmp_path_factory.mktemp("runs")
    cfg = default_config().merged({
        "name": "tiny", "num_clients": 6, "out_dir": str(out),
        "data": {"samples_per_client": 40, "global_test_size": 96, "image_size": 32},
        "train": {"rounds": 2, "local_epochs": 1},
        "selection": {"fraction": 0.5},
    })
    run_experiment(cfg, echo=False)
    return os.path.join(str(out), "tiny"), cfg


def test_runner_writes_every_vehicle_every_round(tiny_run):
    run_dir, cfg = tiny_run
    rounds = data.load_vehicle_rounds(run_dir)
    assert len(rounds) == 2 * cfg.num_clients                   # every car, every round
    assert set(rounds["status"]) <= set(data.VEHICLE_STATUSES)
    reported = rounds[rounds["status"] == "reported"]
    assert reported["alpha"].notna().all()
    # aggregation weights of each round's contributors sum to 1
    assert reported.groupby("round")["alpha"].sum().round(4).eq(1.0).all()
    # per-round "clients_reported" agrees with the per-vehicle records
    agg = data.load_rounds(run_dir).set_index("round")["clients_reported"]
    assert reported.groupby("round").size().reindex(agg.index, fill_value=0).equals(agg)


def test_fleet_table_is_complete_and_consistent(tiny_run):
    run_dir, cfg = tiny_run
    table = data.fleet_table(run_dir)
    assert sorted(table["client_id"]) == list(range(cfg.num_clients))
    assert table["final_accuracy"].between(0, 1).all()
    shares = table.filter(like="share_").sum(axis=1)
    assert shares.round(3).eq(1.0).all()
    rounds = data.load_vehicle_rounds(run_dir)
    live = data.participation_from_rounds(rounds).set_index("client_id")
    final = table.set_index("client_id")
    assert (live["times_reported"] == final["times_reported"]).all()


def test_profiles_rebuilt_from_config_match_written_ones(tiny_run):
    run_dir, _ = tiny_run
    written = data.load_vehicle_profiles(run_dir)
    os.rename(os.path.join(run_dir, "vehicles.json"), os.path.join(run_dir, "v.bak"))
    try:
        rebuilt = data.load_vehicle_profiles(run_dir)
    finally:
        os.rename(os.path.join(run_dir, "v.bak"), os.path.join(run_dir, "vehicles.json"))
    pd.testing.assert_frame_equal(written, rebuilt)
