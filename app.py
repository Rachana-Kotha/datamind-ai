"""
DataMind AI — Streamlit App
Watch a council of AI agents analyze your data in real-time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f0f17; }
[data-testid="stSidebar"] { background: #15151f; border-right: 1px solid #2a2a3a; }

.hero { background: #13131d; border: 1px solid #2a2a3a;
        border-radius: 16px; padding: 36px 40px; margin-bottom: 24px; }
.hero h1 { font-size: 2.2rem; font-weight: 800; color: #e8e6ff;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p  { color: #9d9db8; font-size: 1rem; margin-top: 6px; }

.agent-card { background: #13131d; border: 1px solid #2a2a3a;
              border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
              animation: fadeIn 0.4s ease; }
.agent-card.thinking { border-color: #6d5fcc; }
@keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }

.agent-name { font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.08em; margin-bottom: 4px; }
.agent-msg  { font-size: 0.9rem; color: #c5c3e0; }

.debate-bubble { border-radius: 12px; padding: 12px 16px; margin: 6px 0; font-size: 0.9rem; }
.debate-left  { background: #1e1e2e; border-left: 3px solid #6d5fcc; color: #c5c3e0; }
.debate-right { background: #1a2030; border-left: 3px solid #3b82f6; color: #c5c3e0; }

.metric-box { background: #13131d; border: 1px solid #2a2a3a; border-radius: 10px;
              padding: 16px 20px; text-align: center; }
.metric-val { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-lbl { font-size: 0.75rem; color: #6b6b8a; text-transform: uppercase;
              letter-spacing: 0.05em; margin-top: 2px; }

.risk-high { border-left: 3px solid #ef4444; padding: 10px 14px;
             background: #1f1215; border-radius: 6px; margin: 6px 0; font-size: 0.85rem; color: #fca5a5; }
.risk-medium { border-left: 3px solid #f59e0b; padding: 10px 14px;
               background: #1f1a10; border-radius: 6px; margin: 6px 0; font-size: 0.85rem; color: #fcd34d; }
.risk-low { border-left: 3px solid #10b981; padding: 10px 14px;
            background: #101f17; border-radius: 6px; margin: 6px 0; font-size: 0.85rem; color: #6ee7b7; }
.insight-card { background: #131a25; border: 1px solid #1e3a5f; border-radius: 8px;
                padding: 12px 16px; margin: 6px 0; font-size: 0.85rem; color: #93c5fd; }

.code-block { background: #0d1117; border-radius: 8px; padding: 14px; font-family: monospace;
              font-size: 0.8rem; color: #e2e8f0; overflow-x: auto; white-space: pre; border: 1px solid #21262d; }
.section-header { color: #9d9db8; font-size: 0.75rem; text-transform: uppercase;
                  letter-spacing: 0.1em; margin: 20px 0 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🧠 DataMind AI</h1>
  <p>A council of specialized AI agents analyzes your data — debating findings in real-time, then writing an AI-generated intelligence report.</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "parquet"])

    groq_key = st.text_input(
        "Groq API key (optional — free)",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com — enables AI-written narrative report",
    )
    if groq_key:
        st.success("AI narrative report enabled!")
    else:
        st.info("Add a free Groq key for AI-written reports")

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                preview_df = pd.read_csv(uploaded, nrows=5)
            elif uploaded.name.endswith((".xlsx", ".xls")):
                preview_df = pd.read_excel(uploaded, nrows=5)
            else:
                preview_df = pd.read_parquet(uploaded)
            uploaded.seek(0)

            target_col = st.selectbox("Target column", list(preview_df.columns))
            task_type = st.selectbox("Task type", ["auto", "classification", "regression"])
            run_btn = st.button("Launch council", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            run_btn = False
    else:
        st.info("Upload a CSV to begin")
        run_btn = False

    st.markdown("---")
    st.markdown("### The council")
    for agent_info in [
        ("🔬", "Ada", "EDA Specialist"),
        ("⚡", "Max", "ML Engineer"),
        ("💡", "Iris", "Insight Analyst"),
        ("🔍", "Rex", "Risk Auditor"),
        ("💻", "Cleo", "Code Generator"),
        ("🧠", "Synthesis", "Report Writer"),
    ]:
        em, name, role = agent_info
        st.markdown(f"{em} **{name}** — {role}")

    st.markdown("---")
    st.markdown("**Free stack:** Groq LLaMA 3 · scikit-learn · Streamlit")
    st.markdown("**[⭐ GitHub](https://github.com/YOUR_USERNAME/datamind-ai)**")

# ─── LANDING ────────────────────────────────────────────────────────────────

if not uploaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🤖 Agentic analysis
        Watch 5 AI agents think out loud — EDA, ML, Insights, Critic, Code
        """)
    with col2:
        st.markdown("""
        ### 🗣️ Agent debate
        Agents challenge each other's findings — you see the disagreements
        """)
    with col3:
        st.markdown("""
        ### 📄 AI-written report
        Free Groq LLM synthesizes everything into a narrative report
        """)

    st.markdown("---")
    if st.button("Try with Iris dataset (demo)"):
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.to_csv("/tmp/iris_demo.csv", index=False)
        st.success("Demo dataset saved to /tmp/iris_demo.csv — upload it above!")
    st.stop()

# ─── PREVIEW ────────────────────────────────────────────────────────────────

with st.expander("Dataset preview"):
    st.dataframe(preview_df, use_container_width=True)

if not run_btn:
    st.info("Configure settings in the sidebar and click **Launch council**.")
    st.stop()

# ─── RUN ────────────────────────────────────────────────────────────────────

tmp_path = f"/tmp/datamind_{uploaded.name}"
with open(tmp_path, "wb") as f:
    f.write(uploaded.getvalue())

if tmp_path.endswith(".csv"):
    df = pd.read_csv(tmp_path)
elif tmp_path.endswith((".xlsx", ".xls")):
    df = pd.read_excel(tmp_path)
else:
    df = pd.read_parquet(tmp_path)

if task_type == "auto":
    tgt = df[target_col]
    task_type = "classification" if (tgt.dtype == "object" or tgt.nunique() <= 20) else "regression"

st.markdown("---")
st.markdown("## 🤖 Council in session")
st.caption("Agents are analyzing your data — watch their thoughts below.")

agent_log_container = st.container()
agent_messages = []

def on_agent_thought(agent_name: str, emoji: str, message: str):
    agent_messages.append((agent_name, emoji, message))
    color_map = {"Ada": "#10b981", "Max": "#3b82f6", "Iris": "#f59e0b", "Rex": "#ef4444", "Cleo": "#8b5cf6", "Synthesis": "#a78bfa"}
    color = color_map.get(agent_name, "#9d9db8")
    with agent_log_container:
        st.markdown(f"""
        <div class="agent-card thinking">
          <div class="agent-name" style="color:{color}">{emoji} {agent_name}</div>
          <div class="agent-msg">{message}</div>
        </div>""", unsafe_allow_html=True)

from agents.orchestrator import Orchestrator, AgentMemory
from agents.council import EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent
from agents.synthesis import SynthesisAgent

memory = AgentMemory()
orch = Orchestrator(progress_callback=on_agent_thought)

for AgentCls in [EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent]:
    orch.register_agent(AgentCls(memory=memory, progress_callback=on_agent_thought))

synthesis = SynthesisAgent(memory=memory, groq_api_key=groq_key or None, progress_callback=on_agent_thought)
orch.register_agent(synthesis)

with st.spinner("Council working..."):
    pipeline_result = orch.run(df, target_col, task_type)

findings = pipeline_result["findings"]

st.success(f"✅ Council complete in {pipeline_result['elapsed_seconds']}s!")

# ─── RESULTS TABS ───────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results", "🗣️ Debate", "📄 Report", "💻 Code", "🔍 Risks"])

with tab1:
    st.markdown("### Model leaderboard")
    ada_f = findings.get("Ada", {})
    max_f = findings.get("Max", {})
    iris_f = findings.get("Iris", {})

    shape = ada_f.get("shape", {})
    best_model = max_f.get("best_model", "N/A")
    best_score = max_f.get("best_score", 0)
    metric = max_f.get("metric_name", "score")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{shape.get("rows",0):,}</div><div class="metric-lbl">Rows</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{shape.get("cols",0)}</div><div class="metric-lbl">Columns</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{best_score:.4f}</div><div class="metric-lbl">Best {metric}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{best_model}</div><div class="metric-lbl">Best model</div></div>', unsafe_allow_html=True)

    lb = max_f.get("leaderboard", [])
    if lb:
        st.markdown('<p class="section-header">Leaderboard</p>', unsafe_allow_html=True)
        lb_df = pd.DataFrame([{k: v for k, v in r.items()} for r in lb])
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

    fi = max_f.get("feature_importance")
    if fi:
        st.markdown('<p class="section-header">Top feature importances</p>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({"Feature": fi["features"], "Importance": fi["importances"]})
        st.bar_chart(fi_df.set_index("Feature"), color="#7c3aed")

    insights = iris_f.get("insights", [])
    if insights:
        st.markdown('<p class="section-header">Key insights (Iris)</p>', unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f'<div class="insight-card">💡 <strong>{ins["title"]}</strong><br>{ins["detail"]}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Agent debate")
    st.caption("Watch the agents challenge each other's findings.")
    synth_f = findings.get("Synthesis", {})
    debate = synth_f.get("debate", [])
    for i, entry in enumerate(debate):
        cls = "debate-left" if i % 2 == 0 else "debate-right"
        st.markdown(f'<div class="debate-bubble {cls}"><strong>{entry["emoji"]} {entry["agent"]}</strong><br>{entry["message"]}</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### Intelligence report")
    synth_f = findings.get("Synthesis", {})
    ai_used = synth_f.get("used_ai", False)
    if ai_used:
        st.success("Written by Groq LLaMA 3 (free AI)")
    else:
        st.info("Template report — add a free Groq API key for AI-written narrative")
    narrative = synth_f.get("narrative", "Report unavailable.")
    st.markdown(narrative)

    report_bytes = narrative.encode("utf-8")
    st.download_button("Download report (markdown)", report_bytes, "datamind_report.md", "text/markdown")

with tab4:
    st.markdown("### Generated code snippets (Cleo)")
    cleo_f = findings.get("Cleo", {})
    snippets = cleo_f.get("snippets", {})
    for name, code in snippets.items():
        st.markdown(f'<p class="section-header">{name.replace("_", " ").title()}</p>', unsafe_allow_html=True)
        st.code(code, language="python")

with tab5:
    st.markdown("### Risk audit (Rex)")
    rex_f = findings.get("Rex", {})
    risks = rex_f.get("risks", [])
    caveats = rex_f.get("caveats", [])

    if not risks:
        st.success("No critical risks detected.")
    for r in risks:
        css = f"risk-{r.get('severity', 'low')}"
        st.markdown(f'<div class="{css}"><strong>[{r["severity"].upper()}] {r["risk"]}</strong><br>{r["detail"]}</div>', unsafe_allow_html=True)

    if caveats:
        st.markdown('<p class="section-header">Caveats</p>', unsafe_allow_html=True)
        for c in caveats:
            st.markdown(f"⚠️ {c}")
