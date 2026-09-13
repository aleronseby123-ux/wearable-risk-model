"""
Unit test suite for the Wearable Health Risk Scoring & Underwriting Engine.
Runs with standard library unittest (no external dependencies required).
"""

import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import (
    HealthProfile,
    WearableRiskEngine,
    LogisticRiskClassifier,
    load_kaggle_dataset,
)


class TestWearableModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.csv_path = os.path.join(cls.base_dir, "data", "sleep_health_and_lifestyle_dataset.csv")

    def test_dataset_loading(self):
        """Verify the Kaggle dataset loads cleanly and parses all fields correctly."""
        self.assertTrue(os.path.exists(self.csv_path), f"CSV dataset not found at {self.csv_path}")
        profiles, labels = load_kaggle_dataset(self.csv_path)

        self.assertEqual(len(profiles), 374, "Dataset should contain exactly 374 records")
        self.assertEqual(len(labels), 374, "Labels count must match profile count")

        for p, y in zip(profiles, labels):
            self.assertIn(y, [0, 1], "Target label must be binary 0 or 1")
            self.assertGreater(p.age, 18, "Age should be adult (>18)")
            self.assertGreater(p.sleep_duration, 0.0)
            self.assertGreater(p.daily_steps, 0.0)
            self.assertGreater(p.resting_hr, 40.0)
            self.assertGreater(p.systolic_bp, 70.0)
            self.assertGreater(p.diastolic_bp, 40.0)

    def test_risk_scoring_monotonicity(self):
        """
        Verify the fundamental actuarial premise:
        A sedentary, sleep-deprived applicant (Person A) must receive a significantly
        higher risk score and premium than an active runner with optimal sleep (Person B).
        """
        engine = WearableRiskEngine(base_premium=100.0)

        person_a = HealthProfile(
            age=30,
            sleep_duration=4.5,
            quality_of_sleep=3.0,
            physical_activity_min=15.0,
            stress_level=8.0,
            bmi_category="Overweight",
            resting_hr=85.0,
            daily_steps=2800.0,
            systolic_bp=138.0,
            diastolic_bp=88.0,
        )

        person_b = HealthProfile(
            age=30,
            sleep_duration=8.0,
            quality_of_sleep=9.0,
            physical_activity_min=75.0,
            stress_level=3.0,
            bmi_category="Normal",
            resting_hr=62.0,
            daily_steps=10500.0,
            systolic_bp=118.0,
            diastolic_bp=76.0,
        )

        score_a = engine.assess_profile(person_a)
        score_b = engine.assess_profile(person_b)

        self.assertGreater(score_a.composite_risk_score, score_b.composite_risk_score)
        self.assertGreater(score_a.dynamic_premium, score_b.dynamic_premium)
        self.assertGreaterEqual(score_a.dynamic_premium, 100.0, "High risk applicant should incur a surcharge")
        self.assertLessEqual(score_b.dynamic_premium, 100.0, "Low risk applicant should receive a discount")
        self.assertGreater(score_b.monthly_rebate, score_a.monthly_rebate, "Healthy applicant should earn higher rebates")

    def test_score_boundaries(self):
        """Ensure scores remain within reasonable actuarial bounds [5.0, 98.0]."""
        engine = WearableRiskEngine()

        # Extreme healthy profile
        ultra_healthy = HealthProfile(
            age=25, sleep_duration=8.5, quality_of_sleep=10, physical_activity_min=90,
            stress_level=1, bmi_category="Normal", resting_hr=50, daily_steps=15000,
            systolic_bp=110, diastolic_bp=70
        )
        res_h = engine.assess_profile(ultra_healthy)
        self.assertGreaterEqual(res_h.composite_risk_score, 5.0)
        self.assertLessEqual(res_h.composite_risk_score, 20.0)

        # Extreme unhealthy profile
        ultra_sick = HealthProfile(
            age=65, sleep_duration=3.5, quality_of_sleep=1, physical_activity_min=0,
            stress_level=10, bmi_category="Obese", resting_hr=105, daily_steps=1000,
            systolic_bp=170, diastolic_bp=105
        )
        res_s = engine.assess_profile(ultra_sick)
        self.assertLessEqual(res_s.composite_risk_score, 98.0)
        self.assertGreaterEqual(res_s.composite_risk_score, 80.0)

    def test_classifier_convergence_and_predictions(self):
        """Ensure LogisticRiskClassifier trains, reduces loss, and outputs calibrated probabilities."""
        profiles, labels = load_kaggle_dataset(self.csv_path)

        clf = LogisticRiskClassifier(l2_reg=0.01, learning_rate=0.05, n_epochs=200)
        fit_stats = clf.fit(profiles, labels)

        self.assertLess(fit_stats["final_loss"], fit_stats["initial_loss"], "Optimization must reduce loss")

        # Test evaluation metrics
        metrics = clf.evaluate(profiles, labels)
        self.assertGreater(metrics["accuracy"], 0.85, "Accuracy should exceed 85%")
        self.assertGreater(metrics["f1_score"], 0.85, "F1 score should exceed 0.85")
        self.assertLess(metrics["brier_score"], 0.15, "Brier score should be well-calibrated")

        # Probability outputs must strictly be in [0, 1]
        for p in profiles[:20]:
            prob = clf.predict_proba(p)
            self.assertTrue(0.0 <= prob <= 1.0, f"Probability {prob} out of bounds")


if __name__ == "__main__":
    unittest.main()
