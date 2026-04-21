"""
DataMind AI — Orchestrator
Manages the Council of Agents: dispatches tasks, maintains shared memory,
collects findings, and coordinates the synthesis step.
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# AGENT MEMORY  — shared blackboard all agents read/write
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemory:
    """Shared blackboard that all agents can read and write to."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._log: List[Dict] = []

    def write(self, agent_name: str, key: str, value: Any):
        self._store[f"{agent_name}:{key}"] = value
        self._log.append({
            "agent": agent_name,
            "action": "write",
            "key": key,
            "ts": datetime.now().isoformat(),
        })

    def read(self, agent_name: str, key: str) -> Optional[Any]:
        return self._store.get(f"{agent_name}:{key}")

    def read_all(self) -> Dict[str, Any]:
        return dict(self._store)

    def get_agent_findings(self, agent_name: str) -> Dict:
        return {k.split(":", 1)[1]: v for k, v in self._store.items()
                if k.startswith(f"{agent_name}:")}

    def get_log(self) -> List[Dict]:
        return self._log


# ─────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """All agents inherit from this. Each has a name, role, and personality."""

    def __init__(self, name: str, role: str, emoji: str, personality: str,
                 memory: AgentMemory, progress_callback: Optional[Callable] = None):
        self.name = name
        self.role = role
        self.emoji = emoji
        self.personality = personality
        self.memory = memory
        self.progress_callback = progress_callback
        self.thoughts: List[str] = []
        self.findings: Dict = {}

    def think(self, message: str):
        """Log a thought — shows in the UI as agent reasoning."""
        self.thoughts.append(message)
        if self.progress_callback:
            self.progress_callback(self.name, self.emoji, message)

    def conclude(self, key: str, value: Any):
        """Write a conclusion to shared memory."""
        self.findings[key] = value
        self.memory.write(self.name, key, value)

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    The Orchestrator runs the Council of Agents in sequence.
    Each agent reads from shared memory, adds its own findings,
    and the Synthesis Agent wraps everything into a final report.
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.memory = AgentMemory()
        self.progress_callback = progress_callback
        self.agents: List[BaseAgent] = []
        self.pipeline_log: List[Dict] = []
        self.start_time: Optional[float] = None

    def register_agent(self, agent: BaseAgent):
        self.agents.append(agent)

    def _log(self, event: str, detail: str = ""):
        entry = {"event": event, "detail": detail, "ts": datetime.now().isoformat()}
        self.pipeline_log.append(entry)

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        self.start_time = time.time()
        self._log("pipeline_start", f"Dataset: {df.shape}, target: {target_col}")

        all_findings = {}
        for agent in self.agents:
            self._log(f"agent_start", agent.name)
            try:
                result = agent.run(df, target_col, task_type)
                all_findings[agent.name] = result
                self._log(f"agent_done", agent.name)
            except Exception as e:
                self._log(f"agent_error", f"{agent.name}: {e}")
                all_findings[agent.name] = {"error": str(e)}

        elapsed = round(time.time() - self.start_time, 1)
        self._log("pipeline_done", f"Elapsed: {elapsed}s")

        return {
            "findings": all_findings,
            "memory": self.memory.read_all(),
            "log": self.pipeline_log,
            "elapsed_seconds": elapsed,
        }
