# Contributing to Wearable Health Risk Model

Thank you for your interest in contributing to the Wearable Health Risk Scoring & Underwriting Engine!

## How to Contribute

1. **Fork the repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/wearable-risk-model.git
   cd wearable-risk-model
   ```
3. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
4. **Make your changes** and ensure unit tests pass:
   ```bash
   python3 -m unittest discover tests
   ```
5. **Run the simulation** to verify underwriting consistency:
   ```bash
   python3 src/simulate.py
   ```
6. **Commit and push** your branch, then open a Pull Request.

## Coding Guidelines

- Keep code lightweight, readable, and free of unnecessary heavy external dependencies where Python standard library suffices.
- Add unit tests in `tests/test_model.py` for new scoring logic, models, or data parsers.
- Follow PEP 8 style conventions.
