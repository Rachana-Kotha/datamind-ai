"""
DataMind AI — Synthesis Agent (Generative AI Layer)
Uses free Groq API (LLaMA 3) to write a narrative intelligence report
by synthesizing all agent findings. Falls back to template-based report
if no API key is set — so it ALWAYS works, even without a key.
"""

import os
import json
import time
from typing import Dict, Optional
from agents.orchestrator import BaseAgent, AgentMemory


GROQ_MODEL = "llama3-70b-8192"  # Free on Groq, very capable


class SynthesisAgent(BaseAgent):
    """
    The Council's final voice. Reads all agent findings and uses a free
    Groq LLM to produce a narrative data intelligence report.
    Falls back gracefully if no API key is provided.
    """

    def __init__(self, memory: AgentMemory, groq_api_key: Optional[str] = None,
                 progress_callback=None):
        super().__init__(
            name="Synthesis", role="AI Report Writer", emoji="🧠",
            personality="Articulate, balanced, synthesizes complexity into clarity",
            memory=memory, progress_callback=progress_callback
        )
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

    def run(self, df, target_col: str, task_type: str) -> Dict:
        self.think("Reading all council findings to write the final intelligence report...")

        all_findings = self.memory.read_all()

        # Collect key facts for the LLM prompt
        ada = self.memory.get_agent_findings("Ada")
        max_f = self.memory.get_agent_findings("Max")
        iris = self.memory.get_agent_findings("Iris")
        rex = self.memory.get_agent_findings("Rex")

        shape = ada.get("shape", {})
        missing = ada.get("missing", {})
        correlations = ada.get("correlations", [])
        target_info = ada.get("target_info", {})

        leaderboard = max_f.get("leaderboard", [])
        best_model = max_f.get("best_model", "Unknown")
        best_score = max_f.get("best_score", 0)
        metric_name = max_f.get("metric_name", "score")
        fi = max_f.get("feature_importance")

        insights = iris.get("insights", [])
        risks = rex.get("risks", [])
        caveats = rex.get("caveats", [])

        # Build context summary for LLM
        context = self._build_context(
            shape, target_col, task_type, target_info, missing,
            correlations, leaderboard, best_model, best_score, metric_name,
            fi, insights, risks, caveats
        )

        if self.groq_api_key:
            self.think("Connecting to Groq LLaMA 3 (free AI)... generating narrative report...")
            narrative = self._call_groq(context, target_col, task_type)
        else:
            self.think("No GROQ_API_KEY set — using built-in template report. (Add a free Groq key for AI-written reports!)")
            narrative = self._template_report(
                shape, target_col, task_type, target_info, missing,
                best_model, best_score, metric_name, fi, insights, risks
            )

        # Agent debate summary
        debate = self._generate_debate(insights, risks, best_model, best_score, task_type)

        self.think("Report ready. The council has spoken.")

        findings = {
            "narrative": narrative,
            "debate": debate,
            "executive_summary": self._executive_summary(best_model, best_score, metric_name, shape, insights, risks),
            "context_sent_to_llm": context if self.groq_api_key else None,
            "used_ai": bool(self.groq_api_key),
        }
        for k, v in findings.items():
            self.conclude(k, v)
        return findings

    def _build_context(self, shape, target_col, task_type, target_info,
                       missing, correlations, leaderboard, best_model,
                       best_score, metric_name, fi, insights, risks, caveats):
        ctx = f"""Dataset: {shape.get('rows', 0):,} rows × {shape.get('cols', 0)} columns
Task: {task_type} | Target: '{target_col}'
Target summary: {target_info.get('summary', 'N/A')}

Missing values: {len(missing)} columns affected. Worst: {max(missing, key=lambda c: missing[c]['pct']) + ' (' + str(missing[max(missing, key=lambda c: missing[c]['pct'])]['pct']) + '%)' if missing else 'None'}

Top correlations: {', '.join([f"{c['col1']}↔{c['col2']} (r={c['r']})" for c in correlations[:3]])}

Model leaderboard:
{chr(10).join([f"  {i+1}. {r.get('model', 'Model')} — {metric_name}={r.get(metric_name, r.get('r2', 'N/A'))}, CV={r.get('cv_mean', 'N/A')}±{r.get('cv_std', 'N/A')}" for i, r in enumerate(leaderboard[:3])])}

Best model: {best_model} ({metric_name}={best_score:.4f})

Top features: {', '.join(fi['features'][:5]) if fi else 'N/A'}

Key insights from Iris:
{chr(10).join(['  - ' + i['title'] + ': ' + i['detail'] for i in insights[:4]])}

Risks flagged by Rex:
{chr(10).join(['  - [' + r['severity'].upper() + '] ' + r['risk'] + ': ' + r['detail'] for r in risks[:3]])}

Caveats: {'; '.join(caveats[:2]) if caveats else 'None'}"""
        return ctx

    def _call_groq(self, context: str, target_col: str, task_type: str) -> str:
        try:
            import urllib.request
            import urllib.error

            prompt = f"""You are a senior data scientist writing an intelligence report for a business stakeholder.
Based on this automated analysis of a dataset, write a clear, insightful, and actionable report.

ANALYSIS FINDINGS:
{context}

Write the report in this exact structure (use markdown headers):
## Executive summary
(2-3 sentences on what this data is about and the headline finding)

## What the data tells us
(Explain the key patterns, correlations, and target distribution in plain English)

## Model performance
(Explain which model won, why it likely won, and what the score means in practical terms)

## What drives predictions
(Explain the top features and why they matter)

## Risks and caveats
(Explain the risks Rex flagged, in plain terms a stakeholder can act on)

## Recommended next steps
(3-5 concrete, actionable next steps — be specific)

Write for a smart non-technical audience. Be direct, insightful, and specific. Avoid jargon. 
Total length: 400-600 words."""

            payload = json.dumps({
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.4,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            self.think(f"Groq API call failed ({e}). Falling back to template report.")
            return self._template_report_from_context(context)

    def _template_report(self, shape, target_col, task_type, target_info,
                         missing, best_model, best_score, metric_name,
                         fi, insights, risks) -> str:
        top_features = fi["features"][:3] if fi else ["N/A"]
        risk_text = "\n".join([f"- **[{r['severity'].upper()}]** {r['risk']}: {r['detail']}" for r in risks[:3]]) if risks else "- No critical risks detected."
        insight_text = "\n".join([f"- {i['title']}: {i['detail']}" for i in insights[:4]]) if insights else "- Dataset looks healthy."

        if task_type == "classification":
            rating = "excellent" if best_score > 0.9 else "solid" if best_score > 0.75 else "moderate"
            score_line = f"which is {rating} performance."
        else:
            score_line = f"explaining {best_score:.1%} of variance in the target."

        return f"""## Executive summary

This dataset contains **{shape.get('rows', 0):,} rows** targeting `{target_col}` ({task_type}).
The best performing model is **{best_model}** with a {metric_name} of **{best_score:.4f}**,
{score_line}

## What the data tells us

{f"Missing values were found in {len(missing)} columns and have been handled automatically." if missing else "The dataset is clean with no missing values."}
The most important signals in this dataset come from: **{', '.join(top_features)}**.

## Model performance

The **{best_model}** model emerged as the top performer ({metric_name}: {best_score:.4f}).
Cross-validation confirms this result is robust and not a lucky split.

## What drives predictions

The top predictive features are:
{chr(10).join([f"- **{f}**: importance = {i:.4f}" for f, i in zip((fi['features'][:5] if fi else ['N/A']), (fi['importances'][:5] if fi else [0]))]) }

## Risks and caveats

{risk_text}

## Key insights

{insight_text}

## Recommended next steps

1. **Explore top features** — focus domain knowledge on {top_features[0] if top_features else 'key features'} to understand why it drives predictions.
2. **Handle missing data** — review imputation strategy for the {len(missing)} affected columns.
3. **Try hyperparameter tuning** — use Optuna or GridSearchCV on {best_model} for potential score improvements.
4. **Address any risks** — review the flags from the Critic Agent before deploying.
5. **Monitor in production** — set up data drift detection for key features.

---
*Generated by DataMind AI — Council of Agents*"""

    def _template_report_from_context(self, context: str) -> str:
        return f"""## Auto-generated intelligence report

Based on analysis by the DataMind AI Council of Agents:

{context}

---
*Note: Add a free Groq API key for an AI-written narrative report.*"""

    def _generate_debate(self, insights, risks, best_model, best_score, task_type) -> list:
        """Simulate the agents debating findings — shown in the UI as a chat."""
        debate = []

        debate.append({
            "agent": "Max", "emoji": "⚡",
            "message": f"I'm calling it — {best_model} wins the benchmark with {best_score:.4f}. That's our champion."
        })

        if risks and risks[0]["severity"] in ["high", "critical"]:
            debate.append({
                "agent": "Rex", "emoji": "🔍",
                "message": f"Not so fast, Max. I'm flagging a {risks[0]['severity'].upper()} risk: {risks[0]['risk']}. We can't ignore this."
            })
            debate.append({
                "agent": "Max", "emoji": "⚡",
                "message": f"Fair point, Rex. The score is still good but your concern about '{risks[0]['risk']}' is noted."
            })

        if insights:
            top_insight = insights[0]
            debate.append({
                "agent": "Iris", "emoji": "💡",
                "message": f"What caught my eye: {top_insight['title']}. {top_insight['detail']}"
            })

        debate.append({
            "agent": "Ada", "emoji": "🔬",
            "message": "Agreed. My EDA showed the same underlying pattern. The data is telling a consistent story."
        })

        debate.append({
            "agent": "Cleo", "emoji": "💻",
            "message": "I've generated ready-to-run code for all of this. You can reproduce every finding in minutes."
        })

        debate.append({
            "agent": "Synthesis", "emoji": "🧠",
            "message": "The council has reached a consensus. Report is ready. Let's get this into the hands of the team."
        })

        return debate

    def _executive_summary(self, best_model, best_score, metric_name, shape, insights, risks) -> str:
        high_risks = [r for r in risks if r["severity"] in ["high", "critical"]]
        risk_note = f" ⚠️ {len(high_risks)} high-severity risk(s) flagged." if high_risks else " ✅ No critical risks."
        return (f"Best model: {best_model} ({metric_name}={best_score:.4f}) "
                f"on {shape.get('rows', 0):,} rows.{risk_note} "
                f"{len(insights)} insights found.")
