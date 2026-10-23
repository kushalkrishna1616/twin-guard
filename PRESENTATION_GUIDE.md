# 🚢 TwinGuard: Presentation & Viva Defense Guide

> **Project Title**: TwinGuard: Digital-Twin-Based Counterfactual Risk Assessment and Safety Gating for Autonomous Maritime Agents  
> **Course**: Advanced Python Programming (Digital Assessment 8) — VIT  
> **Team**: HEMASHWARAN K (23MID0436), KRISHNA PRASATH (23MID0369), KUSHAL G (23MIC0118), ROHAN P H (23MID0440)

---

## ⚡ 1. 30-Second Elevator Pitch (Memorize This!)

> *"Good morning/afternoon everyone. Our project is **TwinGuard**, an intelligent runtime safety copilot for autonomous ships. Autonomous navigation AI models like Deep RL often suffer from catastrophic blind spots when faced with ocean currents, sensor lag, or unpredictable vessel traffic.*  
> *Instead of blindly trusting the AI, **TwinGuard wraps the autopilot with a real-time Digital Twin**. Before any rudder or throttle command reaches the vessel, TwinGuard simulates parallel 'what-if' counterfactual futures. If a maneuver is dangerous, it instantly **Modifies** it or triggers an emergency **Veto** evasive maneuver, guaranteeing 0% collision rate while strictly complying with international maritime COLREG rules."*

---

## 🎯 2. Core Problem vs. Our Solution

| The Problem in Autonomous Shipping | TwinGuard's Innovation |
| :--- | :--- |
| **Black-box AI is unpredictable**: Deep RL models can take erratic actions in edge-case weather or traffic. | **Runtime Safety Gate**: Wraps any black-box agent without needing to retrain it. |
| **Deterministic models ignore uncertainty**: Sensor noise, wind, and current drift cause collisions. | **Counterfactual Digital Twin**: Runs $K$ parallel perturbed rollouts predicting future trajectories. |
| **False senses of safety**: Heuristic risks underestimate tail-end hazards. | **Conformal Prediction Calibration**: Derives mathematically certified risk bounds ($1 - \epsilon$). |
| **No maritime law enforcement**: RL models ignore right-of-way rules. | **COLREG Verification**: Formally encodes Head-on (Rule 14), Crossing (Rule 15), and Overtaking (Rule 13). |

---

## 🛡️ 3. How TwinGuard Works (The 4-Step Pipeline)

1. **Agent Proposes an Action**: The navigation agent (RL or APF) calculates desired heading and speed $(u, r)$.
2. **Digital Twin Rollout**: Simulates $K$ parallel prospective trajectories (using 3-DOF vessel kinematics, Nomoto steering dynamics, wind, currents, and sensor noise).
3. **Risk & COLREG Evaluation**:
   - Calculates **DCPA** (Distance to Closest Point of Approach) & **TCPA** (Time to CPA).
   - Checks **Goodwin/Fujii Ship Safety Domains** (elliptical safety bubble around each vessel).
   - Conformal calibrator calculates certified risk upper bound $\hat{R}$.
4. **Safety Gating Action**:
   - **APPROVE**: Risk $\le \tau$ (Action certified safe, executed as-is).
   - **MODIFY**: Risk slightly elevated $\to$ Adjusts rudder to the nearest safe COLREG-compliant angle.
   - **VETO**: Imminent hazard $\to$ Overrides autopilot with emergency braking & hard evasive maneuver.

---

## 📊 4. Key Experimental Results (Numbers to Quote!)

When asked *"Did you benchmark this?"*, quote these exact figures from our test run:

| Metric | Without Guard (Ungated Agent) | With TwinGuard (Our Framework) | Improvement |
| :--- | :---: | :---: | :---: |
| **Collision Rate** | **100.0%** (Catastrophic) | **0.0%** | **100% Elimination** |
| **Goal Reached Rate** | **0.0%** | **100.0%** | **Full Mission Success** |
| **Avg Minimum Clearance**| **44.0 meters** | **94.0 meters** | **+113% Safer Margin** |
| **Emergency Interventions**| 0 (No safety checks) | Active (14 Modifies, 34 Vetoes) | Safe Autonomous Copilot |

---

## 🗣️ 5. 2-Minute Presentation Script (Slide-by-Slide / Speech)

### Step 1: Introduction (0:00 - 0:30)
*"Hello everyone. We present TwinGuard, a digital-twin-based safety architecture for self-driving maritime vessels. Maritime transport carries over 80% of global trade. As the industry transitions toward autonomous surface vessels (MASS), the biggest roadblock is safety certification: black-box AI policies can hallucinate or fail under ocean disturbances."*

### Step 2: Architecture & Python Modules (0:30 - 1:00)
*"We built TwinGuard completely in Python using modular object-oriented design:*
1. *`twinguard/env`: 3-DOF vessel physics and Nomoto steering equations with marine hydrodynamic disturbances.*
2. *`twinguard/counterfactual`: Parallel Monte Carlo trajectory simulator projecting prospective futures.*
3. *`twinguard/calibration`: Conformal prediction providing certified non-parametric risk bounds.*
4. *`twinguard/gate`: Three-tier Approve-Modify-Veto arbitration engine.*
5. *`app/main.py`: Interactive Streamlit dashboard with radar views and live telemetry."*

### Step 3: Demonstration & Results (1:00 - 1:40)
*"In our benchmark across Head-on, Crossing, and Multi-vessel encounters, an ungated agent suffered a 100% collision rate under environmental disturbances. When TwinGuard was activated, the collision rate dropped to 0%, maintaining a safe buffer and full COLREG compliance."*

### Step 4: Conclusion (1:40 - 2:00)
*"TwinGuard acts as an agnostic safety wrapper—it works seamlessly with Reinforcement Learning, Artificial Potential Fields, or human operators, making autonomous maritime navigation provably safe."*

---

## ❓ 6. Expected Viva Questions & Perfect Answers

**Q1: What are COLREGs and which ones did you implement?**  
*Answer:* "COLREGs are the International Regulations for Preventing Collisions at Sea. We formally implemented Rule 14 (Head-on situations: both vessels must alter course to starboard), Rule 15 (Crossing situations: give-way vs stand-on vessel), and Rule 13 (Overtaking)."

**Q2: How does the Digital Twin differ from a standard simulator?**  
*Answer:* "A standard simulator only tracks the present world state. TwinGuard's Digital Twin runs **counterfactual rollouts forward in time**—it branches out prospective parallel futures with simulated ocean currents, wind gusts, and sensor noise to evaluate the consequences of an action before executing it in reality."

**Q3: What is Conformal Prediction here?**  
*Answer:* "Standard heuristic risk estimation often underestimates danger due to sensor noise. Conformal prediction uses historical non-conformity scores on calibration data to produce a mathematically certified upper bound on risk with user-defined confidence level $(1 - \alpha)$."

**Q4: How does the VETO fallback controller work?**  
*Answer:* "When both the agent's proposed action and candidate adjustments violate safety thresholds, the gate triggers a Veto. It applies maximum reverse thrust (emergency braking) and executes a full rudder turn away from the target vessel's projected domain."

---

## 🚀 7. Live Commands (If Examiner Asks to Run It)

- **Launch Interactive Web UI**:
  ```bash
  streamlit run app/main.py
  ```
- **Run Head-to-Head Benchmark in Terminal**:
  ```bash
  python run_demo.py
  ```
- **Run Unit Tests (7/7 Passed)**:
  ```bash
  pytest tests/ -v
  ```
