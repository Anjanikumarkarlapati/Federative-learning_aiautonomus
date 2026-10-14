"""Render captured terminal output as a PNG "screenshot" for docs/run_guide.

    python scripts/render_terminal.py OUT.png "STEP TITLE" "COMMAND" OUTPUT.txt [--max-lines N]

The image shows a terminal window: a title bar with the step name, the prompt line with the
command that was run, and the command's real captured output underneath.
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = (24, 26, 33)
BAR = (44, 47, 58)
FG = (214, 218, 226)
DIM = (140, 146, 160)
PROMPT = (110, 200, 120)
CMD = (240, 240, 240)
ACCENT = (95, 165, 245)
DOTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]


def colour_for(line: str):
    low = line.lower()
    if "passed" in low and "failed" not in low:
        return PROMPT
    if "error" in low or "failed" in low or "traceback" in low:
        return (240, 110, 110)
    if line.lstrip().startswith(("{", "}", "]", "[")) and not line.lstrip().startswith("[ "):
        return DIM
    if line.startswith("[") and "] r" in line:
        return FG
    if line.startswith("|") or line.startswith("#"):
        return ACCENT
    return FG


def render(out: str, title: str, command: str, text: str, max_lines: int, width_chars: int) -> None:
    lines = text.rstrip("\n").splitlines()
    if len(lines) > max_lines:
        keep = max_lines - 1
        lines = lines[:keep] + [f"... ({len(lines) - keep} more lines)"]
    lines = [ln if len(ln) <= width_chars else ln[: width_chars - 1] + "…" for ln in lines]

    size = 15
    font = ImageFont.truetype(MONO, size)
    bold = ImageFont.truetype(MONO_BOLD, size)
    head = ImageFont.truetype(SANS_BOLD, 15)
    cw = font.getbbox("M")[2]
    lh = int(size * 1.45)
    pad = 22
    bar_h = 38

    cmd_lines = [command[i:i + width_chars - 2] for i in range(0, len(command), width_chars - 2)]
    longest = max([len(ln) for ln in lines] + [len(c) + 2 for c in cmd_lines] + [60])
    w = pad * 2 + cw * min(longest, width_chars)
    h = bar_h + pad + lh * (len(cmd_lines) + 1 + len(lines)) + pad

    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, bar_h], fill=BAR)
    for i, c in enumerate(DOTS):
        x = 16 + i * 22
        d.ellipse([x, 13, x + 12, 25], fill=c)
    tw = d.textlength(title, font=head)
    d.text(((w - tw) / 2, 10), title, font=head, fill=FG)

    y = bar_h + pad
    for i, c in enumerate(cmd_lines):
        if i == 0:
            d.text((pad, y), "$ ", font=bold, fill=PROMPT)
        d.text((pad + cw * 2, y), c, font=bold, fill=CMD)
        y += lh
    y += lh // 2
    for ln in lines:
        d.text((pad, y), ln, font=font, fill=colour_for(ln))
        y += lh
    img.save(out, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("title")
    ap.add_argument("command")
    ap.add_argument("text_file")
    ap.add_argument("--max-lines", type=int, default=60)
    ap.add_argument("--width", type=int, default=110)
    a = ap.parse_args()
    with open(a.text_file, encoding="utf-8") as f:
        text = f.read()
    render(a.out, a.title, a.command, text, a.max_lines, a.width)


if __name__ == "__main__":
    main()
