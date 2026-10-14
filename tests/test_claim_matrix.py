"""The patent claim-evidence matrix must only reference code and tests that exist."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_claim_matrix_references_resolve():
    spec = importlib.util.spec_from_file_location(
        "check_claim_matrix", os.path.join(ROOT, "scripts", "check_claim_matrix.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.problems() == []
