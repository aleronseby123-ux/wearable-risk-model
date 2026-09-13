#!/usr/bin/env python3
"""
Interactive CLI Scorer for Wearable Health Risk & Underwriting.
Allows users or underwriters to calculate personalized risk scores,
actuarial rate adjustments, and lifestyle recommendations.

Usage:
  python3 src/cli.py                  # Interactive guided prompt
  python3 src/cli.py --help           # Command-line flags
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model import HealthProfile, WearableRiskEngine, LogisticRiskClassifier, load_kaggle_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wearable Health Risk & Dynamic Underwriting Scorer"
    )
    parser.add_argument("--interactive", action="store_true", default=False, help="Run interactive prompt")
    parser.add_argument("--age", type=int, default=30, help="Applicant age (default: 30)")
    parser.add_argument("--sleep", type=float, default=7.0, help="Average nightly sleep in hours (default: 7.0)")
    parser.add_argument("--quality", type=float, default=7.0, help="Sleep quality rating 1-10 (default: 7.0)")
    parser.add_argument("--steps", type=float, default=7000.0, help="Average daily step count (default: 7000)")
    parser.add_argument("--activity", type=float, default=45.0, help="Daily aerobic activity in minutes (default: 45)")
    parser.add_argument("--hr", type=float, default=70.0, help="Resting heart rate in bpm (default: 70)")
    parser.add_argument("--bp", type=str, default="120/80", help="Blood pressure in mmHg, e.g. 120/80 (default: 120/80)")
    parser.add_argument("--stress", type=float, default=5.0, help="Stress level rating 1-10 (default: 5.0)")
    parser.add_argument("--bmi", type=str, default="Normal", choices=["Normal", "Overweight", "Obese"], help="BMI category")
    return parser.parse_args()


def prompt_user():
    print("\n" + "=" * 65)
    print("  DYNAMIC WEARABLE HEALTH RISK ASSESSMENT")
    print("  Enter your metrics (press Enter to accept default values)")
    print("=" * 65)

    def ask(prompt_text, default, cast_fn):
        try:
            val = input(f"{prompt_text} [{default}]: ").strip()
            return cast_fn(val) if val else default
        except (ValueError, KeyboardInterrupt):
            return default

    age = ask("Applicant Age", 30, int)
    sleep = ask("Nightly Sleep Duration (hours)", 7.0, float)
    qual = ask("Subjective Sleep Quality (1-10)", 7.0, float)
    steps = ask("Daily Step Count", 7000, float)
    act = ask("Physical Activity (min/day)", 45, float)
    hr = ask("Resting Heart Rate (bpm)", 70, float)
    bp = ask("Blood Pressure (systolic/diastolic)", "120/80", str)
    stress = ask("Chronic Stress Index (1-10)", 5.0, float)
    bmi = ask("BMI Category (Normal, Overweight, Obese)", "Normal", str)

    return age, sleep, qual, steps, act, hr, bp, stress, bmi


def main():
    args = parse_args()

    # Determine interactive vs flags
    if args.interactive or (len(sys.argv) == 1 and sys.stdin.isatty()):
        age, sleep, qual, steps, act, hr, bp_str, stress, bmi = prompt_user()
    else:
        age = args.age
        sleep = args.sleep
        qual = args.quality
        steps = args.steps
        act = args.activity
        hr = args.hr
        bp_str = args.bp
        stress = args.stress
        bmi = args.bmi

    parts = bp_str.split("/")
    sys_bp = float(parts[0]) if len(parts) > 0 else 120.0
    dia_bp = float(parts[1]) if len(parts) > 1 else 80.0

    profile = HealthProfile(
        age=age,
        sleep_duration=sleep,
        quality_of_sleep=qual,
        daily_steps=steps,
        physical_activity_min=act,
        resting_hr=hr,
        systolic_bp=sys_bp,
        diastolic_bp=dia_bp,
        stress_level=stress,
        bmi_category=bmi,
    )

    # Optional model prediction
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "sleep_health_and_lifestyle_dataset.csv")

    ml_prob = None
    if os.path.exists(csv_path):
        try:
            profiles, labels = load_kaggle_dataset(csv_path)
            clf = LogisticRiskClassifier(n_epochs=250)
            clf.fit(profiles, labels)
            ml_prob = clf.predict_proba(profile)
        except Exception:
            pass

    engine = WearableRiskEngine(base_premium=100.0)
    res = engine.assess_profile(profile, ml_prob=ml_prob)

    print("\n" + "=" * 65)
    print(f"  ACTUARIAL TELEMATICS ASSESSMENT REPORT (AGE {age})")
    print("=" * 65)
    print(f"Composite Health Risk Score : {res.composite_risk_score:5.1f} / 100")
    print(f"Underwriting Risk Tier      : {res.risk_tier}")
    if ml_prob is not None:
        print(f"ML Clinical Risk Probability: {res.ml_probability * 100:5.1f}%")
    print("-" * 65)
    print(f"Traditional Static Premium  : ${res.static_premium:6.2f} / month ($1,200.00 / yr)")
    print(f"Dynamic Wearable Premium    : ${res.dynamic_premium:6.2f} / month (${res.dynamic_premium * 12:,.2f} / yr)")
    sign = "+" if res.premium_delta_pct >= 0 else ""
    print(f"Premium Adjustment          : {sign}{res.premium_delta_pct:.1f}% vs Static Intake Form")
    print(f"Monthly Vitality Rebate     : ${res.monthly_rebate:6.2f} / month")
    print("-" * 65)
    print("Additive Points Contribution Breakdown:")
    for name, pts in res.contributions.items():
        print(f"  - {name:28s}: {pts:+5.1f} pts")
    print("-" * 65)
    print("Actionable Habit Recommendations:")
    for rec in res.recommendations:
        print(f"  • {rec}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
