#!/usr/bin/env python3
"""Audit exported tracks for numeric defects and abrupt sensor-frame jumps.

These are consistency checks, not ground-truth tracking or speed measurements.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(directory):
    records = json.loads((directory / "tracks.json").read_text())
    last, seen, steps, issues = {}, set(), [], []
    for record in records:
        key = (record["frame"], record["id"])
        if key in seen:
            issues.append(
                dict(
                    type="duplicate_id_in_frame", frame=record["frame"], id=record["id"]
                )
            )
        seen.add(key)
        values = [
            record[key] for key in ("time", "x", "y", "vx_relative", "vy_relative")
        ]
        if not np.isfinite(values).all():
            issues.append(
                dict(type="nonfinite_output", frame=record["frame"], id=record["id"])
            )
            continue
        previous = last.get(record["id"])
        if previous:
            dt = record["time"] - previous["time"]
            if dt <= 0:
                issues.append(
                    dict(
                        type="nonmonotonic_track_time",
                        frame=record["frame"],
                        id=record["id"],
                    )
                )
            elif dt <= 0.5:
                distance = float(
                    np.hypot(record["x"] - previous["x"], record["y"] - previous["y"])
                )
                steps.append(distance / dt)
                if distance > 1.0 + 30.0 * dt:
                    issues.append(
                        dict(
                            type="outside_demo_motion_envelope",
                            frame=record["frame"],
                            id=record["id"],
                            distance_m=distance,
                            dt_s=dt,
                        )
                    )
        last[record["id"]] = record
    return dict(
        observations=len(records),
        issue_count=len(issues),
        issues=issues,
        short_gap_steps=len(steps),
        implied_relative_speed_mps=(
            dict(
                zip(
                    ("p50", "p95", "max"),
                    map(float, np.percentile(steps, [50, 95, 100])),
                )
            )
            if steps
            else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="results/ai_validation_v2")
    parser.add_argument("--after", default="results/robustness_validation")
    parser.add_argument("--output", default="results/robustness_audit.json")
    args = parser.parse_args()
    report = dict(
        warning="Output consistency only; sensor frame, approximate calibration, no ground truth.",
        envelope=dict(position_slack_m=1.0, max_relative_speed_mps=30.0, window_s=0.5),
        bags={},
    )
    for bag in ("studentProject", "studentProject1"):
        report["bags"][bag] = {
            label: audit(Path(root) / bag)
            for label, root in (("before", args.before), ("after", args.after))
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if any(result["after"]["issue_count"] for result in report["bags"].values()):
        raise SystemExit("Runtime audit failed; inspect the saved report.")


if __name__ == "__main__":
    main()
