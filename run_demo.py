"""
TwinGuard Command-Line Demo Runner.
Runs an end-to-end evaluation comparing Ungated vs TwinGuard Gated agents.
"""

import sys
import numpy as np
import pandas as pd

from twinguard.env.maritime_env import MaritimeEnv
from twinguard.scenarios.encounters import create_head_on_scenario, create_crossing_scenario
from twinguard.agents.rule_based import APFMaritimeAgent
from twinguard.gate.gate import TwinGuardSafetyGate
from twinguard.benchmark.evaluator import BenchmarkRunner


def main():
    print("=" * 70)
    print("TwinGuard: Smart Safety Copilot for Self-Driving Ships")
    print("=" * 70)

    # 1. Single Head-on Encounter Demonstration
    print("\n[Step 1] Running Head-on Encounter Simulation...")
    print("Scenario: Two ships are heading directly toward each other.")
    print("Traffic Rule: Both vessels must steer right to pass safely.")
    
    scen = create_head_on_scenario(distance=800.0)
    env = MaritimeEnv(scenario_config=scen, enable_disturbances=True)
    agent = APFMaritimeAgent(colreg_aware=False, name="Ship_Autopilot")
    gate = TwinGuardSafetyGate(safety_threshold=0.35, num_rollouts=8, horizon_steps=20)
    runner = BenchmarkRunner(gate=gate)

    print("\nRunning simulation with TwinGuard Safety Guard active...")
    metrics, traj, audit = runner.run_episode(env=env, agent=agent, is_gated=True)

    print(f"\nJourney Outcome:")
    print(f"  - Total Timesteps:      {metrics.total_steps}")
    print(f"  - Closest Clearance:    {metrics.min_distance:.1f} meters")
    print(f"  - Collision Occurred:   {metrics.collision}")
    print(f"  - Safe Passing Buffer:  {'Respected' if not metrics.domain_violation else 'Breached'}")
    print(f"  - Copilot Corrections:  {metrics.interventions_total} times ({metrics.interventions_modify} Adjustments, {metrics.interventions_veto} Emergency Overrides)")

    print("\nSample Decision Log (What the Copilot Did):")
    for r in audit[:4]:
        dec_name = r['decision'].value if hasattr(r['decision'], 'value') else str(r['decision'])
        print(f"  Second {r['step']:02d} | Action: {dec_name:<7} | Risk: {r['calibrated_upper_risk']:.2f} | Reason: {r['reason']}")

    # 2. Quick Comparative Test
    print("\n" + "=" * 70)
    print("[Step 2] Head-to-Head Comparison: Without Guard vs With TwinGuard")
    print("=" * 70)

    suite = runner.run_comparative_benchmark(
        agent=agent,
        scenarios=["head_on", "crossing_give_way"],
        episodes_per_scenario=1
    )

    df_summary = suite.to_summary_dataframe()
    print("\nResults Summary:")
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("Demo completed successfully.")
    print("To open the interactive visual dashboard, run: streamlit run app/main.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
