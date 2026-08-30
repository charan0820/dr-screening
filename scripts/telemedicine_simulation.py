"""
Owner: P4
Day 5: lightweight queueing simulation for the teleophthalmology capacity
story (spec Sections 40-43), scoped down to plain Python/SimPy instead of
MATLAB Simulink, per the hackathon scoping decision.

Models a simple pipeline:
    patients arrive -> image acquisition -> quality check (some need retake)
    -> AI processing -> uncertain/referable cases go to ophthalmologist queue
    -> ophthalmologist review

This answers RQ10 at hackathon scale: given N ophthalmologists and a review
time per case, what daily patient volume can the system handle before the
ophthalmologist queue becomes the bottleneck?

Usage:
    python scripts/telemedicine_simulation.py

Produces:
    results/telemedicine_capacity.png
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_scenario(
    patients_per_day: int,
    ungradable_rate: float = 0.10,
    referable_or_uncertain_rate: float = 0.25,
    ophthalmologist_review_minutes: float = 0.5,  # 30 sec target from spec Section 26
    num_ophthalmologists: int = 1,
    working_hours_per_day: float = 8.0,
) -> dict:
    """
    Simple deterministic capacity model (not a full discrete-event
    simulation — that's the honest scope for a 6-day hackathon prototype;
    a real SimPy/Simulink model would add arrival-time variance and queueing
    delay distributions, which is future work, not claimed here).
    """
    gradable_patients = patients_per_day * (1 - ungradable_rate)
    cases_needing_review = gradable_patients * referable_or_uncertain_rate

    available_minutes_per_day = working_hours_per_day * 60 * num_ophthalmologists
    minutes_needed = cases_needing_review * ophthalmologist_review_minutes

    utilization = minutes_needed / available_minutes_per_day if available_minutes_per_day > 0 else float("inf")
    max_reviewable_cases = available_minutes_per_day / ophthalmologist_review_minutes
    bottlenecked = cases_needing_review > max_reviewable_cases

    return {
        "patients_per_day": patients_per_day,
        "cases_needing_review": round(cases_needing_review, 1),
        "ophthalmologist_utilization": round(utilization, 3),
        "max_reviewable_cases_per_day": round(max_reviewable_cases, 1),
        "bottlenecked": bottlenecked,
    }


def find_required_ophthalmologists(patients_per_day: int, **kwargs) -> int:
    """Minimum ophthalmologist count to keep utilization <= 1.0 for this volume."""
    n = 1
    while True:
        result = simulate_scenario(patients_per_day, num_ophthalmologists=n, **kwargs)
        if result["ophthalmologist_utilization"] <= 1.0:
            return n
        n += 1
        if n > 500:  # safety cap
            return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    scenarios = [100, 500, 1000, 100000 // 365 + 1]  # last one: 100k+/year -> per-day estimate
    scenario_labels = ["100/day", "500/day", "1,000/day", "~274/day (100k/yr)"]

    print("Scenario analysis (1 ophthalmologist, 30s review time, 8hr day):\n")
    results = []
    for n_patients, label in zip(scenarios, scenario_labels):
        r = simulate_scenario(n_patients, num_ophthalmologists=1)
        results.append(r)
        status = "BOTTLENECKED" if r["bottlenecked"] else "OK"
        print(f"  {label}: {r['cases_needing_review']} cases need review, "
              f"utilization={r['ophthalmologist_utilization']:.1%} [{status}]")

    print("\nMinimum ophthalmologists required to avoid bottleneck at each volume:")
    required_counts = []
    for n_patients, label in zip(scenarios, scenario_labels):
        req = find_required_ophthalmologists(n_patients)
        required_counts.append(req)
        print(f"  {label}: {req} ophthalmologist(s)")

    os.makedirs(args.outdir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(scenario_labels, required_counts, color="#4a3267")
    ax1.set_ylabel("Ophthalmologists required")
    ax1.set_title("Staffing to Avoid Bottleneck")
    ax1.tick_params(axis="x", rotation=15)

    utilizations = [r["ophthalmologist_utilization"] * 100 for r in results]
    ax2.plot(scenario_labels, utilizations, marker="o", color="#de638a", linewidth=2)
    ax2.set_ylabel("Ophthalmologist utilization (%)")
    ax2.set_title("Workload Scaling (1 Ophthalmologist)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.set_ylim(0, max(utilizations) * 1.3)

    plt.tight_layout()
    out_path = os.path.join(args.outdir, "telemedicine_capacity.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    with open(os.path.join(args.outdir, "telemedicine_scenarios.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(
        "\n[NOTE] This is a simplified deterministic capacity model (average-case "
        "math), not a full discrete-event queueing simulation with arrival-time "
        "variance — an honest scope for a 6-day prototype. Framed this way in "
        "the demo/report, not as a full Simulink-equivalent."
    )


if __name__ == "__main__":
    main()
