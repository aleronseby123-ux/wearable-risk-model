# Wearable Health Risk Scoring & Dynamic Insurance Underwriting Engine

[![CI](https://github.com/aleronseby123-ux/wearable-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/aleronseby123-ux/wearable-risk-model/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](data/sleep_health_and_lifestyle_dataset.csv)


A proof-of-concept predictive risk-scoring and dynamic underwriting model demonstrating how ongoing wearable and biometric telematics solve the **"Static Intake Form Paradox"** in life and health insurance.

---

## 1. The Core Problem: The Two 30-Year-Olds Paradox

Traditional health and life insurance underwriting is a **single-afternoon snapshot**:
- Two 30-year-olds both apply for a policy and check: *"Non-smoker, no pre-existing chronic conditions, healthy history"*.
- **Person A**: Sleeps 4.5 hours a night, takes 2,800 steps/day (sedentary), experiences chronic high stress, has a resting heart rate of 85 bpm and blood pressure of 138/88 mmHg.
- **Person B**: Sleeps 8.0 hours a night, runs daily (10,500 steps/day), has a resting heart rate of 62 bpm and blood pressure of 118/76 mmHg.

### Under Traditional Static Underwriting:
Both applicants are lumped into the exact same underwriting risk class and charged the exact same monthly premium (e.g., **\$100.00 / month**).
The insurer inadvertently forces Person B to subsidize Person A's hidden, escalating clinical mortality and morbidity hazard.

### Under Dynamic Wearable Underwriting:
Continuous telematics instantly reveal the divergence:
- **Person A**: Composite Risk Score **98.0 / 100** (High Risk / Substandard Tier). Premium adjusted to **\$165.00 / month** (+65.0% risk surcharge).
- **Person B**: Composite Risk Score **5.0 / 100** (Preferred Plus Tier). Premium discounted to **\$75.00 / month** (-25.0% preferred discount) plus an optional **\$25.00 / month** Vitality cash-back rebate.
- **Resulting Underwriting Spread**: **\$90.00 / month** (\$1,080.00 / year).

---

## 2. Dataset Provenance & Key Findings

We utilize the public **Sleep Health and Lifestyle Dataset** from Kaggle:
- **Sample Size**: 374 subjects aged 27–59 across multiple professions.
- **Wearable & Biometric Variables**:
  - `Sleep Duration`: Continuous hours of sleep per night (5.8 – 8.5h in dataset, 4.0 – 10.0h in model domain).
  - `Quality of Sleep`: Subjective and sensor-derived sleep restoration score (1 – 10).
  - `Physical Activity Level`: Daily aerobic activity duration (minutes/day).
  - `Daily Steps`: Accelerometer step count (3,000 – 10,000 steps).
  - `Heart Rate`: Resting heart rate (bpm).
  - `Blood Pressure`: Systolic and diastolic arterial pressure (mmHg).
  - `Stress Level`: Allostatic and psychological stress index (1 – 10).
  - `BMI Category`: Normal, Overweight, Obese.
- **Clinical Adverse Target**: Confirmed clinical diagnosis of Insomnia or Sleep Apnea, or Stage 1/2 Hypertension ($\text{BP} \ge 130/85$ mmHg or $\text{HR} \ge 82$ bpm).

---

## 3. Mathematical & Machine Learning Architecture

### A. Domain-Calibrated Risk Engine (`WearableRiskEngine`)
Computes an explainable, bounded 0–100 Health Risk Score:
$$\text{Score} = \max\Big(5.0, \min\big(98.0, \, S_{\text{base}}(\text{Age}) + S_{\text{sleep}} + S_{\text{activity}} + S_{\text{cardio}} + S_{\text{stress}}\big)\Big)$$

1. **Sleep Deficit Penalty ($S_{\text{sleep}}$)**:
   - Penalties calibrated to epidemiological literature: severe penalty for $<5.0$h (+26 pts) and $<6.0$h (+18 pts); quality bonus/malus.
2. **Sedentary / Activity Penalty ($S_{\text{activity}}$)**:
   - $<3,500$ steps/day: +22 pts (high cardiovascular hazard).
   - $\ge 10,000$ steps/day: -8 pts longevity credit.
3. **Cardiovascular & Autonomic Strain ($S_{\text{cardio}}$)**:
   - Resting HR $\ge 84$ bpm (+16 pts); resting HR $\le 65$ bpm (-4 pts).
   - AHA Blood Pressure staging: Stage 2 (+20 pts), Stage 1 (+12 pts), Normal (-2 pts).
4. **Metabolic & Stress Load ($S_{\text{stress}}$)**:
   - Chronic stress $\ge 8/10$ (+10 pts) and Obesity (+16 pts).

### B. Machine Learning Predictive Classifier (`LogisticRiskClassifier`)
A pure Python, L2-regularized logistic regression model trained on 80% of the dataset and validated on 20% holdout:
- **Optimization**: Mini-batch Gradient Descent with inverse epoch learning rate schedule.
- **Objective**: Minimized binary cross-entropy with Ridge regularization ($L_2 = 0.02$).
- **Test Performance**:
  - **Accuracy**: 97.3%
  - **Precision**: 97.3%
  - **Recall**: 97.3%
  - **F1 Score**: 0.973
  - **Brier Calibration Score**: 0.0349

### C. Dynamic Actuarial Pricing & Behavioral Rebates
$$\text{Premium}_{\text{dynamic}} = \text{Premium}_{\text{static}} \times M(S)$$
- **Preferred Plus** ($S < 28$): Multiplier $0.75$ (\$75/mo), up to \$25/mo monthly cash-back rebate.
- **Standard Healthy** ($28 \le S < 42$): Multiplier $0.90$ (\$90/mo).
- **Elevated Risk** ($42 \le S < 60$): Multiplier $1.25$ (\$125/mo).
- **High Risk / Substandard** ($S \ge 60$): Multiplier $1.65$ (\$165/mo).

---

## 4. Quickstart & Execution

No third-party packages or complex installations required; runs with standard Python 3.

### 1. Run the Command-Line Simulation & Case Study
```bash
python3 src/simulate.py
```
This executes:
1. Dataset ingestion and 80/20 train/test evaluation.
2. The side-by-side comparison of the Two 30-Year-Olds.
3. A 90-day longitudinal coaching trajectory showing Person A cutting their premium from \$165/mo to \$75/mo.

### 2. Run the Interactive Personalized Risk Scorer
Score any individual or input your own wearable metrics:
```bash
# Interactive guided prompt
python3 src/cli.py

# Or pass custom metrics directly via flags
python3 src/cli.py --age 30 --sleep 8.0 --steps 10000 --hr 62 --bp 118/76
```

### 3. Run the Automated Unit Test Suite
```bash
python3 -m unittest discover tests
```

### 4. Launch the Interactive Web Dashboard
Open `src/dashboard.html` in your web browser:
```bash
# On macOS:
open src/dashboard.html
```
Features:
- **Real-Time Sliders**: Adjust sleep, steps, resting HR, blood pressure, and stress.
- **Live Gauge & Risk Tier**: 0–100 score indicator updates instantly.
- **Waterfall Contribution Bars**: Shows exactly which habit drives points up or down.
- **Interactive 90-Day Roadmap**: Step through Day 1 to Day 90 to see habits translate into dollar savings.

---

## 5. File Structure

```
wearable-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml                             # Automated testing on multi-Python matrix
├── data/
│   └── sleep_health_and_lifestyle_dataset.csv  # 374 Kaggle telemetry records
├── src/
│   ├── model.py                               # Core scoring engine & logistic classifier
│   ├── simulate.py                            # CLI case study & validation runner
│   ├── cli.py                                 # Interactive personalized CLI risk scorer
│   └── dashboard.html                         # Interactive visual web dashboard
├── tests/
│   └── test_model.py                          # Automated unit test suite
├── CONTRIBUTING.md                            # Contribution guidelines
├── LICENSE                                    # MIT License
└── README.md                                  # Complete documentation
```

---

## 6. License & Data Citation

- **Code License**: This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
- **Data Source**: Telemetry data sourced from Kaggle's public *Sleep Health and Lifestyle Dataset* (anonymized public health benchmark).

