#!/usr/bin/env python3
import argparse
import os
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from env.articulated_action_mask import (  # noqa: E402
    default_action_mask_path,
    generate_sweep_tables,
    save_sweep_tables,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the offline ZL50GN articulated dual-body sweep table."
    )
    parser.add_argument("--output", default=default_action_mask_path())
    parser.add_argument("--trace-samples", type=int, default=8)
    args = parser.parse_args()

    start = time.perf_counter()
    tables = generate_sweep_tables(trace_samples=max(1, args.trace_samples))
    output = save_sweep_tables(args.output, tables)
    elapsed = time.perf_counter() - start
    print(
        "saved {} | shape={} | {:.2f}s".format(
            output,
            tables["sweep_table_front"].shape,
            elapsed,
        )
    )


if __name__ == "__main__":
    main()
