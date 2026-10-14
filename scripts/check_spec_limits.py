"""Check the Form 2 draft against IP India fee and format limits.

    python scripts/check_spec_limits.py [docs/patent/FORM2_PROVISIONAL.md]

* abstract <= 150 words, starts with the title, cites a figure (Rule 13(7))
* claims <= 10 (more cost INR 320 each for natural persons, as reported)
* pages <= 30 (more cost INR 160 each); estimated at 350 words per A4 page of
  1.5-spaced 12 pt text, plus one page per drawing sheet
* every reference numeral used in the text is defined in REFERENCE_NUMERALS.md
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS_PER_PAGE = 350


def section(text: str, heading: str) -> str:
    m = re.search(rf"^## {heading}\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def check(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    issues = []
    title = section(text, "TITLE").splitlines()[0].strip() if section(text, "TITLE") else ""
    abstract = section(text, "ABSTRACT")
    words = len(re.findall(r"\b[\w'-]+\b", abstract))
    if words > 150:
        issues.append(f"abstract has {words} words (> 150)")
    if title and not abstract.lower().startswith(title.lower()[:40]):
        issues.append("abstract does not start with the title")
    if not re.search(r"FIG\.\s*\d", abstract):
        issues.append("abstract cites no figure")
    claims = re.findall(r"^\s*(\d+)\.\s", section(text, "CLAIMS"), re.M)
    if len(claims) > 10:
        issues.append(f"{len(claims)} claims (> 10)")
    body_words = len(re.findall(r"\b\w+\b", text))
    drawings = os.path.join(ROOT, "docs", "patent", "drawings")
    sheets = len([f for f in os.listdir(drawings) if f.endswith(".png")]) if os.path.isdir(drawings) else 0
    pages = body_words / WORDS_PER_PAGE + sheets
    if pages > 30:
        issues.append(f"about {pages:.0f} pages (> 30)")
    with open(os.path.join(ROOT, "docs", "patent", "REFERENCE_NUMERALS.md"), encoding="utf-8") as fh:
        known = set(re.findall(r"^\| (\d+) \|", fh.read(), re.M))
    used = set(re.findall(r"\((\d{3})\)", text)) | set(re.findall(r"\((\d{3}),", text))
    missing = sorted(used - known)
    if missing:
        issues.append(f"numerals not in REFERENCE_NUMERALS.md: {', '.join(missing)}")
    print(f"abstract {words} words · {len(claims)} claims · ~{pages:.1f} pages "
          f"({body_words} words + {sheets} drawing sheets)")
    return issues


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "patent", "FORM2_PROVISIONAL.md")
    found = check(target)
    for issue in found:
        print("FAIL:", issue)
    print("limits OK" if not found else f"{len(found)} issue(s)")
    sys.exit(1 if found else 0)
