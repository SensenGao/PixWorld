#!/usr/bin/env python
"""Run the self-tests.

    python train/selftest.py            # all of them
    python train/selftest.py cameras    # just one

``render`` needs a GPU; the other three run on CPU.
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TESTS = {
    "cameras": "geometry.cameras",
    "schedule": "diffusion.schedule",
    "trajectories": "geometry.trajectories",
    "render": "geometry.render",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", choices=list(TESTS) + [[]], default=[],
                   help="which to run; default is all")
    args = p.parse_args()
    names = args.names or list(TESTS)

    failed = []
    for name in names:
        print(f"\n=== {name} ({TESTS[name]}) ===", flush=True)
        try:
            runpy.run_module(TESTS[name], run_name="__main__")
        except BaseException as e:                                   # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"FAILED: {type(e).__name__}: {e}", flush=True)

    print()
    if failed:
        for n, e in failed:
            print(f"FAILED  {n}: {e}")
        raise SystemExit(1)
    print(f"all {len(names)} self-tests OK")


if __name__ == "__main__":
    main()
