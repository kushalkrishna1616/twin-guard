"""
Benchmark Evaluator: Compares Gated vs Ungated Agents across safety,
COLREG compliance, and operational efficiency metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from twinguard.env.maritime_env import MaritimeEnv
from twinguard.scenarios.encounters import ScenarioConfig, SCENARIO_BUILDERS
from twinguard.agents.base import BaseAgent
from twinguard.gate.gate import TwinGuardSafetyGate, GateDecision


@dataclass
class EpisodeMetrics:
    """Performance metrics for a single simulation episode."""
    scenario_name: str
    is_gated: bool
    collision: bool
    domain_violation: bool
    colreg_violation: bool
    goal_reached: bool
    min_distance: float
    total_steps: int
    total_reward: float
    interventions_total: int
    interventions_modify: int
    interventions_veto: int
    path_length: float


@dataclass
class BenchmarkSuiteResult:
    """Aggregated comparison across multiple evaluation runs."""
    ungated_metrics: list[EpisodeMetrics]
    gated_metrics: list[EpisodeMetrics]

    def to_summary_dataframe(self) -> pd.DataFrame:
        ungated_df = pd.DataFrame([asdict(m) for m in self.ungated_metrics])
        gated_df = pd.DataFrame([asdict(m) for m in self.gated_metrics])

        summary_rows = []
        for name, df in [("Ungated Agent", ungated_df), ("TwinGuard (Gated)", gated_df)]:
            n = len(df)
            if n == 0:
                continue
            summary_rows.append({
                "Configuration": name,
                "Episodes": n,
                "Collision Rate (%)": round(100.0 * df["collision"].mean(), 1),
                "Domain Violation Rate (%)": round(100.0 * df["domain_violation"].mean(), 1),
                "COLREG Violation Rate (%)": round(100.0 * df["colreg_violation"].mean(), 1),
                "Goal Reached Rate (%)": round(100.0 * df["goal_reached"].mean(), 1),
                "Avg Min Clearance (m)": round(df["min_distance"].mean(), 1),
                "Avg Interventions/Episode": round(df["interventions_total"].mean(), 2),
                "Avg Steps": round(df["total_steps"].mean(), 1)
            })

        return pd.DataFrame(summary_rows)


class BenchmarkRunner:
    """
    Executes controlled comparisons between Gated and Ungated configurations.
    """
    def __init__(self, gate: TwinGuardSafetyGate | None = None):
        self.gate = gate or TwinGuardSafetyGate()

    def run_episode(
        self,
        env: MaritimeEnv,
        agent: BaseAgent,
        is_gated: bool = True
    ) -> tuple[EpisodeMetrics, list[dict], list[dict]]:
        """
        Runs one full episode.
        Returns (metrics, trajectory_history, audit_records).
        """
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        path_length = 0.0
        prev_pos = np.array(env.own_ship.position)

        interventions_mod = 0
        interventions_veto = 0
        audit_history: list[dict] = []

        if is_gated:
            self.gate.reset()

        while not (terminated or truncated):
            # Agent proposes black-box action
            raw_action = agent.predict(obs, info)

            if is_gated:
                exec_action, audit_record = self.gate.evaluate_and_gate(
                    own_ship=env.own_ship,
                    target_ships=env.target_ships,
                    proposed_action=raw_action
                )
                audit_history.append(asdict(audit_record))
                if audit_record.decision == GateDecision.MODIFY:
                    interventions_mod += 1
                elif audit_record.decision == GateDecision.VETO:
                    interventions_veto += 1
            else:
                exec_action = raw_action

            obs, reward, terminated, truncated, info = env.step(exec_action)
            total_reward += reward

            curr_pos = np.array(env.own_ship.position)
            path_length += float(np.linalg.norm(curr_pos - prev_pos))
            prev_pos = curr_pos

        metrics = EpisodeMetrics(
            scenario_name=env.scenario_config.name,
            is_gated=is_gated,
            collision=bool(info.get("collision", False)),
            domain_violation=bool(info.get("domain_violation", False)),
            colreg_violation=bool(info.get("colreg_violation", False)),
            goal_reached=bool(info.get("goal_reached", False)),
            min_distance=round(float(info.get("min_distance", 0.0)), 1),
            total_steps=env.current_step,
            total_reward=round(total_reward, 1),
            interventions_total=interventions_mod + interventions_veto,
            interventions_modify=interventions_mod,
            interventions_veto=interventions_veto,
            path_length=round(path_length, 1)
        )

        return metrics, env.trajectory_history, audit_history

    def run_comparative_benchmark(
        self,
        agent: BaseAgent,
        scenarios: list[str] | None = None,
        episodes_per_scenario: int = 3
    ) -> BenchmarkSuiteResult:
        """
        Executes head-to-head comparison across standard scenarios.
        """
        scenario_keys = scenarios or ["head_on", "crossing_give_way", "overtaking", "multi_vessel"]
        ungated_results: list[EpisodeMetrics] = []
        gated_results: list[EpisodeMetrics] = []

        for key in scenario_keys:
            builder = SCENARIO_BUILDERS[key]
            for ep in range(episodes_per_scenario):
                # 1. Run Ungated
                env_ungated = MaritimeEnv(scenario_config=builder())
                m_ungated, _, _ = self.run_episode(env_ungated, agent, is_gated=False)
                ungated_results.append(m_ungated)

                # 2. Run Gated
                env_gated = MaritimeEnv(scenario_config=builder())
                m_gated, _, _ = self.run_episode(env_gated, agent, is_gated=True)
                gated_results.append(m_gated)

        return BenchmarkSuiteResult(
            ungated_metrics=ungated_results,
            gated_metrics=gated_results
        )
