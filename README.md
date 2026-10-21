# TwinGuard: Digital-Twin-Based Counterfactual Risk Assessment and Safety Gating for Autonomous Maritime Agents

Advanced Python Programming (Digital Assessment 8)  
Vellore Institute of Technology (VIT)

Team Members:
- HEMASHWARAN K (23MID0436)
- KRISHNA PRASATH (23MID0369)
- KUSHAL G (23MIC0118)
- ROHAN P H (23MID0440)

---

## 1. Overview
TwinGuard is a runtime safety verification framework for autonomous maritime vessels. Autonomous navigation models (such as Reinforcement Learning agents) can make hazardous decisions when encountering sensor noise, delayed perception, or erratic traffic from other vessels.

TwinGuard wraps any black-box navigation agent with a digital twin simulation layer. Before an action is executed, it runs K prospective counterfactual ("what-if") rollouts to calculate collision risk, ship domain violations, and maritime COLREG rule compliance.

- APPROVE: If the action is certified safe across rollouts.
- MODIFY: If the action is slightly unsafe, adjusts steering or speed to the nearest safe action.
- VETO: If no safe action exists, overrides the agent with a safe fallback controller (emergency braking and evasive turning).

---

## 2. Key Modules
1. twinguard/env/: 3-DOF vessel kinematics, Nomoto steering dynamics, wind, ocean currents, and sensor noise.
2. twinguard/scenarios/: Standard maritime encounter generators (Head-on Rule 14, Crossing Rule 15, Overtaking Rule 13, and multi-vessel).
3. twinguard/risk/: Distance and Time to Closest Point of Approach (DCPA/TCPA), Goodwin/Fujii ship safety domains, and COLREG verification.
4. twinguard/counterfactual/: Rollout engine simulating K parallel what-if futures with environmental and sensor perturbations.
5. twinguard/calibration/: Conformal prediction framework generating certified risk upper bounds.
6. twinguard/gate/: Approve, Modify, and Veto decision logic with fallback controller.
7. twinguard/explainability/: Decision attribution and human-readable audit trail.
8. twinguard/benchmark/: Comparative evaluation suite comparing Gated vs Ungated agents.
9. app/main.py: Clean Streamlit user interface with live simulation, rollout fans, and evaluation charts.

---

## 3. How to Run

### Step 1: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 2: Run Automated Tests
```bash
pytest tests/test_twinguard.py -v
```

### Step 3: Run the Command-Line Demo
To run an instant simulation and comparative benchmark directly in the terminal:
```bash
python run_demo.py
```

### Step 4: Run the Interactive Web Dashboard
```bash
streamlit run app/main.py
```
Open your browser and navigate to: http://localhost:8501

---

## 4. Evaluation Summary
The framework measures:
- Collision Rate (%)
- Safety Domain Violation Rate (%)
- COLREG Compliance (%)
- Mission Completion Rate (%)
- Gate Intervention Frequency (Modifications and Vetoes)
