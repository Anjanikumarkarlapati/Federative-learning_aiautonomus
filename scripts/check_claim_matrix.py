"""Verify that every code and test reference in the claim-evidence matrix exists.

    python scripts/check_claim_matrix.py      # exit 1 and list what is missing

A reference is a backticked ``path::Name`` or ``path::Class.method``; the file
must exist and define that function/class/method (or, for tests, that test).
A backticked bare path must exist on disk (globs allowed for results).
"""
from __future__ import annotations

import ast
import glob
import os
import re
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "docs", "patent", "CLAIM_EVIDENCE_MATRIX.md")


def defined_names(path: str) -> set:
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}.{item.name}")
    return names


def problems() -> List[str]:
    with open(MATRIX, encoding="utf-8") as fh:
        text = fh.read()
    out = []
    for ref in re.findall(r"`([^`]+)`", text):
        if "::" in ref:
            path, name = ref.split("::", 1)
            full = os.path.join(ROOT, path)
            if not os.path.exists(full):
                out.append(f"missing file: {path}")
            elif name not in defined_names(full):
                out.append(f"missing definition: {ref}")
        elif "/" in ref and not ref.startswith("rounds"):
            if not glob.glob(os.path.join(ROOT, ref)):
                out.append(f"missing path: {ref}")
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print(p)
    print("claim matrix OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
