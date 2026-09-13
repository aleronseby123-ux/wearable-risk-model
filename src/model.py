"""
wearable-risk-model: Core Modeling Engine
Includes:
1. WearableRiskEngine: Transparent actuarial/domain-engineered 0-100 risk scoring.
2. LogisticRiskClassifier: Supervised ML classifier with L2 regularization trained on Kaggle data.
3. ActuarialPricingModel: Converts wearable risk scores into adjusted insurance premiums & rebates.
"""

import math
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class HealthProfile:
    """Individual health & wearable telemetry profile."""
    person_id: Optional[str] = None
    age: int = 30
    gender: str = "Male"
    occupation: str = "Office Worker"
    sleep_duration: float = 7.0         # Hours per night
    quality_of_sleep: float = 7.0       # 1 to 10
    physical_activity_min: float = 45.0 # Minutes per day
    stress_level: float = 5.0           # 1 to 10
    bmi_category: str = "Normal"        # Normal, Overweight, Obese
    resting_hr: float = 70.0            # Beats per minute
    daily_steps: float = 7000.0         # Step count
    systolic_bp: float = 120.0          # mmHg
    diastolic_bp: float = 80.0          # mmHg
    sleep_disorder: str = "None"        # None, Insomnia, Sleep Apnea


@dataclass
class RiskAssessment:
    """Detailed risk assessment output."""
    composite_risk_score: float         # 0 (optimal) to 100 (severe risk)
    risk_tier: str                      # Preferred, Standard, Elevated, High Risk
    subscores: Dict[str, float]         # Component breakdown
    contributions: Dict[str, float]     # Additive point contributions
    ml_probability: float               # Predicted probability of adverse clinical outcome
    static_premium: float               # Traditional snapshot premium ($/mo)
    dynamic_premium: float              # Wearable-adjusted premium ($/mo)
    premium_delta_pct: float            # Percentage change vs static
    monthly_rebate: float               # Vitality-style incentive rebate ($/mo)
    recommendations: List[str]          # Actionable habit targets


class WearableRiskEngine:
    """
    Actuarial & Biometric Telematics Risk Scoring Engine.
    Converts continuous wearable telemetry into a standardized 0-100 risk score
    calibrated against clinical thresholds and empirical health risks.
    """

    def __init__(self, base_premium: float = 100.0):
        self.base_premium = base_premium

    def compute_sleep_penalty(self, duration: float, quality: float) -> Tuple[float, List[str]]:
        """
        Sleep deficit scoring based on clinical sleep medicine guidelines:
        - 7.0 - 8.5 hours is optimal.
        - < 6.0 hours: chronic sleep restriction (elevates cortisol, BP, insulin resistance).
        - < 5.0 hours: severe acute/chronic deficit.
        """
        penalty = 0.0
        recs = []

        if duration < 5.0:
            penalty += 26.0
            recs.append("Critical sleep deprivation (<5h/night): Major driver of cardiovascular strain.")
        elif duration < 6.0:
            penalty += 18.0
            recs.append("Sub-optimal sleep (<6h/night): Increases hypertension risk.")
        elif duration < 6.8:
            penalty += 8.0
        elif duration > 9.5:
            penalty += 6.0  # Hypersomnia / sedentary marker
        else:
            penalty += 0.0  # Optimal sleep

        # Quality multiplier (1-10)
        if quality <= 4.0:
            penalty += 10.0
            recs.append("Poor sleep fragmentation / quality rating.")
        elif quality <= 6.0:
            penalty += 5.0
        elif quality >= 8.0:
            penalty -= 3.0  # Restorative sleep bonus

        return max(0.0, penalty), recs

    def compute_activity_penalty(self, steps: float, active_min: float) -> Tuple[float, List[str]]:
        """
        Activity & Sedentary scoring:
        - > 9,000 steps: Highly active (significant longevity bonus)
        - 7,000 - 9,000 steps: Moderate / healthy baseline
        - 4,500 - 7,000 steps: Low active
        - < 4,500 steps: Sedentary lifestyle
        """
        penalty = 0.0
        recs = []

        if steps < 3500:
            penalty += 22.0
            recs.append("Severely sedentary (<3,500 steps/day): 2.4x higher all-cause mortality hazard.")
        elif steps < 5500:
            penalty += 14.0
            recs.append("Low step count (<5,500 steps/day): Increase daily walking to lower base premium.")
        elif steps < 7500:
            penalty += 6.0
        elif steps >= 10000:
            penalty -= 8.0  # Active longevity credit
        elif steps >= 8000:
            penalty -= 4.0

        # Physical activity duration (min/day)
        if active_min < 25:
            penalty += 8.0
            recs.append("Aerobic activity under 25 min/day.")
        elif active_min >= 60:
            penalty -= 4.0

        return penalty, recs

    def compute_cardiovascular_penalty(self, resting_hr: float, sys_bp: float, dia_bp: float) -> Tuple[float, List[str]]:
        """
        Autonomic & Cardiovascular biometrics:
        - Resting HR: Elevated resting heart rate (>75-80 bpm) indicates poor stroke volume & sympathetic hyperactivity.
        - Blood Pressure: AHA / ACC staging.
        """
        penalty = 0.0
        recs = []

        # Resting Heart Rate
        if resting_hr >= 84:
            penalty += 16.0
            recs.append(f"Elevated resting heart rate ({resting_hr:.0f} bpm): Tachycardia / autonomic strain.")
        elif resting_hr >= 78:
            penalty += 9.0
            recs.append(f"Borderline elevated resting heart rate ({resting_hr:.0f} bpm).")
        elif resting_hr <= 65:
            penalty -= 4.0  # Good aerobic efficiency bonus

        # Blood pressure
        if sys_bp >= 140 or dia_bp >= 90:
            penalty += 20.0
            recs.append(f"Stage 2 Hypertension ({sys_bp:.0f}/{dia_bp:.0f} mmHg): Immediate clinical follow-up warranted.")
        elif sys_bp >= 130 or dia_bp >= 80:
            penalty += 12.0
            recs.append(f"Stage 1 Hypertension ({sys_bp:.0f}/{dia_bp:.0f} mmHg): Moderate vascular resistance.")
        elif sys_bp >= 120 and dia_bp < 80:
            penalty += 5.0
            recs.append(f"Elevated blood pressure ({sys_bp:.0f}/{dia_bp:.0f} mmHg).")
        else:
            penalty -= 2.0  # Optimal BP

        return penalty, recs

    def compute_metabolic_stress_penalty(self, bmi_cat: str, stress: float) -> Tuple[float, List[str]]:
        """Metabolic & Allostatic load (BMI & Chronic Stress Index)."""
        penalty = 0.0
        recs = []

        cat = bmi_cat.strip().lower()
        if "obese" in cat:
            penalty += 16.0
            recs.append("Obesity category: Elevates sleep apnea, diabetes, and cardiovascular hazard.")
        elif "overweight" in cat:
            penalty += 8.0

        if stress >= 8.0:
            penalty += 10.0
            recs.append("High chronic stress index (≥8/10): Contributes to sleep disruption & BP surges.")
        elif stress >= 6.0:
            penalty += 5.0
        elif stress <= 3.0:
            penalty -= 3.0

        return penalty, recs

    def assess_profile(self, p: HealthProfile, ml_prob: Optional[float] = None) -> RiskAssessment:
        """
        Evaluate full wearable profile, calculating composite 0-100 score,
        underwriting tier, and actuarial pricing adjustment.
        """
        # Baseline anchor for age 30 = 20 points
        base_anchor = 15.0 + max(0.0, (p.age - 25) * 0.6)

        s_sleep, r_sleep = self.compute_sleep_penalty(p.sleep_duration, p.quality_of_sleep)
        s_act, r_act = self.compute_activity_penalty(p.daily_steps, p.physical_activity_min)
        s_cardio, r_cardio = self.compute_cardiovascular_penalty(p.resting_hr, p.systolic_bp, p.diastolic_bp)
        s_meta, r_meta = self.compute_metabolic_stress_penalty(p.bmi_category, p.stress_level)

        total_points = base_anchor + s_sleep + s_act + s_cardio + s_meta
        # Bound score between 5 (elite athlete) and 98 (critical risk)
        score = max(5.0, min(98.0, total_points))

        # Risk Tier Classification
        if score < 28.0:
            tier = "Preferred Plus (Lowest Risk)"
            multiplier = 0.75  # 25% discount
            rebate = 25.0
        elif score < 42.0:
            tier = "Standard Healthy"
            multiplier = 0.90  # 10% discount
            rebate = 15.0
        elif score < 60.0:
            tier = "Moderate / Elevated Risk"
            multiplier = 1.25  # 25% surcharge
            rebate = 5.0
        else:
            tier = "High Risk / Substandard"
            multiplier = 1.65  # 65% surcharge
            rebate = 0.0

        static_prem = self.base_premium
        dynamic_prem = round(static_prem * multiplier, 2)
        pct_delta = round(((dynamic_prem - static_prem) / static_prem) * 100.0, 1)

        recommendations = r_sleep + r_act + r_cardio + r_meta
        if not recommendations:
            recommendations.append("Excellent wearable metrics! Maintain current routine to keep Preferred tier.")

        return RiskAssessment(
            composite_risk_score=round(score, 1),
            risk_tier=tier,
            subscores={
                "sleep": round(s_sleep, 1),
                "activity": round(s_act, 1),
                "cardiovascular": round(s_cardio, 1),
                "metabolic_stress": round(s_meta, 1),
            },
            contributions={
                "Baseline Age Anchor": round(base_anchor, 1),
                "Sleep Duration & Quality": round(s_sleep, 1),
                "Activity & Daily Steps": round(s_act, 1),
                "Resting HR & Blood Pressure": round(s_cardio, 1),
                "BMI & Stress Level": round(s_meta, 1),
            },
            ml_probability=round(ml_prob if ml_prob is not None else 0.0, 3),
            static_premium=round(static_prem, 2),
            dynamic_premium=dynamic_prem,
            premium_delta_pct=pct_delta,
            monthly_rebate=rebate,
            recommendations=recommendations,
        )


class LogisticRiskClassifier:
    """
    Pure Python regularized Logistic Regression classifier.
    Trained on Kaggle Sleep Health & Lifestyle telemetry to predict
    probability of adverse clinical health conditions (Hypertension / Sleep Disorders).
    """

    def __init__(self, l2_reg: float = 0.01, learning_rate: float = 0.05, n_epochs: int = 400):
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.means: List[float] = []
        self.stds: List[float] = []
        self.feature_names = [
            "Sleep Duration",
            "Quality of Sleep",
            "Physical Activity (min)",
            "Stress Level",
            "BMI Weight Index",
            "Resting Heart Rate",
            "Daily Steps (k)",
            "Systolic BP",
            "Diastolic BP"
        ]

    def _extract_features(self, p: HealthProfile) -> List[float]:
        bmi_num = 0.0
        cat = p.bmi_category.lower()
        if "obese" in cat:
            bmi_num = 2.0
        elif "overweight" in cat:
            bmi_num = 1.0

        return [
            float(p.sleep_duration),
            float(p.quality_of_sleep),
            float(p.physical_activity_min),
            float(p.stress_level),
            bmi_num,
            float(p.resting_hr),
            float(p.daily_steps) / 1000.0,
            float(p.systolic_bp),
            float(p.diastolic_bp),
        ]

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-18.0, min(18.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, profiles: List[HealthProfile], labels: List[int]) -> Dict[str, Any]:
        """
        Train regularized logistic model via gradient descent.
        labels: 1 if high clinical risk (disorder or hypertension), 0 if healthy.
        """
        n_samples = len(profiles)
        if n_samples == 0:
            raise ValueError("No training samples provided.")

        raw_X = [self._extract_features(p) for p in profiles]
        n_features = len(raw_X[0])

        # Compute means and stds for feature standardization
        self.means = [0.0] * n_features
        self.stds = [0.0] * n_features
        for j in range(n_features):
            col = [raw_X[i][j] for i in range(n_samples)]
            m = sum(col) / n_samples
            var = sum((x - m) ** 2 for x in col) / n_samples
            s = math.sqrt(var) if var > 1e-6 else 1.0
            self.means[j] = m
            self.stds[j] = s

        # Standardize X
        X = []
        for row in raw_X:
            norm_row = [(row[j] - self.means[j]) / self.stds[j] for j in range(n_features)]
            X.append(norm_row)

        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Gradient Descent loop
        losses = []
        for epoch in range(self.n_epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            total_loss = 0.0

            for i in range(n_samples):
                xi = X[i]
                yi = labels[i]
                linear = sum(self.weights[j] * xi[j] for j in range(n_features)) + self.bias
                y_hat = self._sigmoid(linear)

                eps = 1e-12
                loss_i = -(yi * math.log(max(eps, y_hat)) + (1 - yi) * math.log(max(eps, 1 - y_hat)))
                total_loss += loss_i

                diff = y_hat - yi
                for j in range(n_features):
                    grad_w[j] += diff * xi[j]
                grad_b += diff

            lr = self.learning_rate / (1.0 + 0.001 * epoch)
            for j in range(n_features):
                grad = (grad_w[j] / n_samples) + (self.l2_reg * self.weights[j])
                self.weights[j] -= lr * grad
            self.bias -= lr * (grad_b / n_samples)

            reg_penalty = 0.5 * self.l2_reg * sum(w ** 2 for w in self.weights)
            losses.append((total_loss / n_samples) + reg_penalty)

        return {
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "weights": {self.feature_names[j]: round(self.weights[j], 4) for j in range(n_features)},
            "bias": round(self.bias, 4),
        }

    def predict_proba(self, p: HealthProfile) -> float:
        """Predict probability P(Risk=1)."""
        raw_feat = self._extract_features(p)
        norm_feat = [(raw_feat[j] - self.means[j]) / self.stds[j] for j in range(len(raw_feat))]
        z = sum(self.weights[j] * norm_feat[j] for j in range(len(norm_feat))) + self.bias
        return self._sigmoid(z)

    def evaluate(self, profiles: List[HealthProfile], labels: List[int]) -> Dict[str, float]:
        """Compute evaluation metrics: Accuracy, Precision, Recall, F1, and Brier Score."""
        preds = [self.predict_proba(p) for p in profiles]
        binary_preds = [1 if prob >= 0.5 else 0 for prob in preds]

        tp = sum(1 for p, y in zip(binary_preds, labels) if p == 1 and y == 1)
        tn = sum(1 for p, y in zip(binary_preds, labels) if p == 0 and y == 0)
        fp = sum(1 for p, y in zip(binary_preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(binary_preds, labels) if p == 0 and y == 1)

        total = len(labels)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (fn + tp) if (fn + tp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        brier = sum((prob - y) ** 2 for prob, y in zip(preds, labels)) / total if total > 0 else 0.0

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "n_samples": total,
            "positives": sum(labels),
            "negatives": total - sum(labels)
        }


def load_kaggle_dataset(csv_path: str) -> Tuple[List[HealthProfile], List[int]]:
    """
    Parse Kaggle Sleep Health and Lifestyle dataset into HealthProfiles and clinical risk targets.
    Clinical target = 1 if subject has Sleep Apnea, Insomnia, or Hypertension (BP >= 130/80 or HR >= 82).
    """
    profiles: List[HealthProfile] = []
    labels: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Person ID"):
                continue

            bp = row.get("Blood Pressure", "120/80").split("/")
            sys_bp = float(bp[0]) if len(bp) > 0 else 120.0
            dia_bp = float(bp[1]) if len(bp) > 1 else 80.0

            disorder = row.get("Sleep Disorder", "None").strip()
            hr = float(row.get("Heart Rate", 70))
            is_hypertensive = (sys_bp >= 130.0 or dia_bp >= 85.0 or hr >= 82.0)
            has_disorder = (disorder in ["Sleep Apnea", "Insomnia"])

            target = 1 if (has_disorder or is_hypertensive) else 0

            p = HealthProfile(
                person_id=row.get("Person ID"),
                gender=row.get("Gender", "Unknown"),
                age=int(row.get("Age", 30)),
                occupation=row.get("Occupation", "Unknown"),
                sleep_duration=float(row.get("Sleep Duration", 7.0)),
                quality_of_sleep=float(row.get("Quality of Sleep", 7.0)),
                physical_activity_min=float(row.get("Physical Activity Level", 45.0)),
                stress_level=float(row.get("Stress Level", 5.0)),
                bmi_category=row.get("BMI Category", "Normal"),
                resting_hr=hr,
                daily_steps=float(row.get("Daily Steps", 6000.0)),
                systolic_bp=sys_bp,
                diastolic_bp=dia_bp,
                sleep_disorder=disorder
            )

            profiles.append(p)
            labels.append(target)

    return profiles, labels
