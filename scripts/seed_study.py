#!/usr/bin/env python
"""
Offline robustness study.

Answers the two questions a technical judge should ask:

  1. Does the detector actually alert on the slow-drift attack, or did we get
     lucky with one seed?
  2. Is the comparison against the fixed threshold fair, or did we simply pick
     a threshold that could never fire?

Run:  .venv/bin/python scripts/seed_study.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from qtwin.pipeline import calibrate_false_alarms, run_simulation  # noqa: E402
from qtwin.scenarios import SCENARIO_ORDER, SCENARIOS  # noqa: E402

N_SEEDS = 200


def main() -> None:
    print("=" * 74)
    print("FALSE-ALARM CALIBRATION  (clean channel, nothing happening)")
    print("=" * 74)
    cal = calibrate_false_alarms(n_runs=N_SEEDS)
    print(f"{cal.n_runs} independent clean runs of {cal.n_rounds} rounds each\n")
    for k, v in cal.false_alarm_rate.items():
        first = cal.mean_first_alarm[k]
        extra = f"  (mean first alarm round {first:.0f})" if first else ""
        print(f"  {k:20s} {v:6.1%}{extra}")

    print()
    print("=" * 74)
    print("DETECTION PERFORMANCE PER SCENARIO")
    print("=" * 74)
    for key in SCENARIO_ORDER:
        ours, matched, leads, missed = [], [], [], 0
        for s in range(60000, 60000 + N_SEEDS):
            r = run_simulation(key, seed=s)
            if r.alert_cusum is None:
                missed += 1
                continue
            ours.append(r.alert_cusum)
            m = r.alert_static["matched"]
            matched.append(m)
            leads.append(np.inf if m is None else m - r.alert_cusum)
        onset = SCENARIOS[key].onset
        print(f"\n  {SCENARIOS[key].name}  (onset {onset})")
        if key == "clean":
            print(f"    alarms raised on a clean channel: "
                  f"{len(ours)}/{N_SEEDS}")
            continue
        finite = [x for x in leads if np.isfinite(x)]
        never = len(leads) - len(finite)
        print(f"    detected            {len(ours)}/{N_SEEDS} runs")
        print(f"    our alert round     median {np.median(ours):.0f}  "
              f"[{min(ours)}-{max(ours)}]   "
              f"delay after onset {np.median(ours) - (onset or 0):.0f}")
        if finite:
            print(f"    lead over matched   median {np.median(finite):.0f} "
                  f"rounds; positive in {np.mean(np.array(finite) > 0):.0%} "
                  f"of runs")
        print(f"    matched never fired {never} run(s)")

    print()
    print("=" * 74)
    print("Interpretation: the lead is a distribution, not a single number.")
    print("Quote the median and the spread to judges, never one lucky run.")
    print("=" * 74)


if __name__ == "__main__":
    main()
