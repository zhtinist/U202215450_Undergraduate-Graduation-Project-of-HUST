#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# 仓库根：本文件位于 src/experiment_tools/cplmp/，向上 3 级到 HaotianZhu
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
LEC = SRC / "LecSlides_370K"

for p in (ROOT, SRC, LEC, LEC / "train", ROOT / "log" / "repro" / "active" / "scripts"):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

from cplmp_runtime_patch import apply_cplmp_patch


def main() -> None:
    apply_cplmp_patch()
    from train import train

    train()


if __name__ == "__main__":
    main()
