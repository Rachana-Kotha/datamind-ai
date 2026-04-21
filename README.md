# 🧠 DataMind AI — Council of Agents

**The first open-source multi-agent data intelligence system.**  
Watch 5 specialized AI agents analyze your data, debate each other's findings,  
then synthesize everything into an AI-written intelligence report — all for free.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![Groq LLaMA 3](https://img.shields.io/badge/LLM-Groq%20LLaMA%203-purple)](https://groq.com)

---

## 🎬 What makes this unique

Most AutoML tools give you a leaderboard. DataMind AI gives you a **council**.

Five agents with distinct personalities run in parallel, read each other's findings through shared memory, then *debate* — and you watch it happen in real-time.

| Agent | Name | Personality | What they do |
|---|---|---|---|
| 🔬 | **Ada** | Methodical, thorough | Deep EDA — nulls, distributions, correlations |
| ⚡ | **Max** | Competitive, benchmark-obsessed | Trains & ranks 3+ models with CV |
| 💡 | **Iris** | Creative, connects dots | Extracts business-level insights |
| 🔍 | **Rex** | Skeptical, rigorous | Audits for risks, leakage, biases |
| 💻 | **Cleo** | Pragmatic, ships code | Generates ready-to-run Python snippets |
| 🧠 | **Synthesis** | Articulate, balanced | Uses free Groq LLaMA 3 to write the report |

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/datamind-ai.git
cd datamind-ai
pip install -r requirements.txt

# Web app (recommended)
streamlit run app.py

# CLI — demo mode
python run.py --demo

# CLI — your dataset
python run.py --file titanic.csv --target Survived
```

**Optional (for AI-written reports):** Get a free API key at [console.groq.com](https://console.groq.com) — no credit card needed.

```bash
# Set your free Groq key
export GROQ_API_KEY=gsk_...
streamlit run app.py
```

---

## 🏗️ Architecture

```
datamind-ai/
├── app.py                    # Streamlit UI (dark mode, live agent thoughts)
├── run.py                    # CLI interface
├── requirements.txt
├── agents/
│   ├── orchestrator.py       # AgentMemory blackboard + Orchestrator
│   ├── council.py            # Ada, Max, Iris, Rex, Cleo agents
│   └── synthesis.py          # Synthesis agent (Groq LLM + fallback)
└── sample_data/
    └── sample_titanic.csv
```

### How agents share knowledge

Every agent reads from and writes to a shared **AgentMemory** blackboard:

```python
# Ada writes her findings
memory.write("Ada", "correlations", [...])

# Iris reads Ada's work
correlations = memory.read("Ada", "correlations")

# Rex reads both Ada and Max
eda = memory.get_agent_findings("Ada")
ml  = memory.get_agent_findings("Max")
```

This means each agent *knows what the others found* — enabling the debate.

---

## 🤖 GenAI + Agentic AI features

### Agentic pipeline
- Agents are dispatched by an **Orchestrator** with a shared goal
- Each agent has a **defined role**, **personality**, and **tool access**
- Agents communicate through **shared memory** (blackboard pattern)
- The pipeline is **fully automated** — no human in the loop

### Generative AI (Groq LLaMA 3)
- The Synthesis Agent sends a structured prompt to **LLaMA 3-70B** (free via Groq)
- The LLM writes a **narrative intelligence report** from all agent findings
- Falls back to a template report if no API key is set — **always works**

### Agent debate
- Max claims a winner → Rex challenges it → Iris adds context → Ada confirms → Cleo ships code
- Shown as a **chat-style debate** in the UI

---

## 📊 What you get

| Output | Description |
|---|---|
| Live agent thoughts | Watch each agent reason in real-time |
| Model leaderboard | Ranked by accuracy/R² with cross-validation |
| Feature importance | Top predictive features with importances |
| Agent debate | Agents argue over findings |
| AI narrative report | LLM-written plain-English report (with Groq key) |
| Risk audit | Leakage, imbalance, overfitting flags |
| Code snippets | Ready-to-run Python for every step |

---

## 🛠️ Use it as a Python library

```python
from agents.orchestrator import Orchestrator, AgentMemory
from agents.council import EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent
from agents.synthesis import SynthesisAgent
import pandas as pd

df = pd.read_csv("my_data.csv")
memory = AgentMemory()

def log(name, emoji, msg):
    print(f"{emoji} [{name}] {msg}")

orch = Orchestrator(progress_callback=log)
for Cls in [EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent]:
    orch.register_agent(Cls(memory=memory, progress_callback=log))
orch.register_agent(SynthesisAgent(memory=memory))  # no key = template report

result = orch.run(df, target_col="Survived", task_type="classification")

print(result["findings"]["Synthesis"]["narrative"])
print(result["findings"]["Max"]["best_model"])
```

---

## 🆓 100% Free Stack

| Component | Free service |
|---|---|
| ML models | scikit-learn (MIT license) |
| LLM (narrative) | Groq API — free tier, LLaMA 3-70B |
| Web UI | Streamlit — free |
| Hosting | Streamlit Community Cloud — free |
| Everything else | Python stdlib |

**No paid APIs. No credit card. No limits.**

---

## 🚀 Deploy for free in 5 minutes

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → deploy
4. Add `GROQ_API_KEY` in Streamlit secrets for AI reports

---

## 🗺️ Roadmap

- [ ] LangGraph-based agent graph (true parallel execution)
- [ ] Agent memory persistence (SQLite)
- [ ] Custom agent plugins
- [ ] SHAP explainability
- [ ] Time series support
- [ ] Hugging Face model hub integration

---

## 🤝 Contributing

PRs welcome! Each agent is a self-contained class — easy to extend.

---

## 📄 License

MIT — free for personal and commercial use.

---

Built with ❤️ using Python · scikit-learn · Streamlit · Groq LLaMA 3
