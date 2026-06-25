#!/usr/bin/env python3
from __future__ import annotations

"""
Standalone eval entry with runtime CPLMP patch.
This file keeps GitHub baseline source untouched.
"""

from cplmp_runtime_patch import apply_cplmp_patch

# Import baseline evaluator after patch helper is available.
import eval_udop_t5_10pct_baseline as base_eval


def main() -> None:
    apply_cplmp_patch()
    base_eval.main()


if __name__ == "__main__":
    main()
