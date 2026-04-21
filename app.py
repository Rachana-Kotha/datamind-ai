"""
DataMind AI — Full App (Updated)
Adds: KPI storytelling · PDF/DOCX/MD export · Share URL · Email tracking · User capture
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, io, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="DataMind AI", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f0f17; }
[data-testid="stSidebar"] { background: #15151f; border-right: 1px solid #2a2a3a; }
.hero { background: #13131d; border: 1px solid #2a2a3a; border-radius: 16px;
        padding: 36px 40px; margin-bottom: 24px; }
.hero h1 { font-size: 2.2rem; font-weight: 800; color: #e8e6ff; margin: 0; }
.hero p { color: #9d9db8; font-size: 1rem; margin-top: 6px; }
.kpi-card { background: #13131d; border: 1px solid #6d5fcc; border-radius: 12px;
            padding: 20px 18px; margin-bottom: 10px; }
.kpi-val { font-size: 2rem; font-weight: 800; color: #a78bfa; }
.kpi-lbl { font-size: 0.72rem; color: #6b6b8a; text-transform: uppercase;
           letter-spacing: 0.08em; }
.kpi-story { font-size: 0.88rem; color: #c5c3e0; margin-top: 10px; line-height: 1.6; }
.kpi-trend-up   { color: #34d399; font-weight: 700; }
.kpi-trend-down { color: #f87171; font-weight: 700; }
.kpi-trend-flat { color: #9d9db8; font-weight: 700; }
.agent-card { background: #13131d; border: 1px solid #2a2a3a; border-radius: 12px;
              padding: 14px 18px; margin-bottom: 10px; animation: fadeIn 0.4s ease; }
.agent-name { font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.08em; margin-bottom: 4px; }
.agent-msg { font-size: 0.9rem; color: #c5c3e0; }
.debate-bubble { border-radius: 12px; padding: 12px 16px; margin: 6px 0; font-size: 0.9rem; }
.debate-left  { background: #1e1e2e; border-left: 3px solid #6d5fcc; color: #c5c3e0; }
.debate-right { background: #1a2030; border-left: 3px solid #3b82f6; color: #c5c3e0; }
.metric-box { background: #13131d; border: 1px solid #2a2a3a; border-radius: 10px;
              padding: 16px 20px; text-align: center; }
.metric-val { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-lbl { font-size: 0.75rem; color: #6b6b8a; text-transform: uppercase;
              letter-spacing: 0.05em; margin-top: 2px; }
.risk-high   { border-left: 3px solid #ef4444; padding: 10px 14px; background: #1f1215;
               border-radius: 6px; margin: 6px 0; font-size: 0.85rem; color: #fca5a5; }
.risk-medium { border-left: 3px solid #f59e0b; padding: 10px 14px; background: #1f1a10;
               border-radius: 6px; margin: 6px 0; font-size: 0.85rem; color: #fcd34d; }
.insight-card { background: #131a25; border: 1px solid #1e3a5f; border-radius: 8px;
                padding: 12px 16px; margin: 6px 0; font-size: 0.85rem; color: #93c5fd; }
.share-box { background: #1a1a2e; border: 2px solid #6d5fcc; border-radius: 12px;
             padding: 20px 24px; margin: 16px 0; }
.user-form { background: #13131d; border: 1px solid #2a2a3a; border-radius: 12px;
             padding: 24px; margin: 16px 0; }
@keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
</style>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 DataMind AI</h1>
  <p>A council of AI agents analyzes your data, generates 5 KPIs with stories, and creates a shareable intelligence report — PDF, Word, or web link.</p>
</div>
""", unsafe_allow_html=True)

# ─── CHECK FOR SHARED REPORT LINK ────────────────────────────────────────────
query_params = st.query_params
shared_token = query_params.get("report", None)

if shared_token:
    from src.notifications import decode_share_token, send_notification_email
    metadata = decode_share_token(shared_token)
    if metadata:
        st.info(f"📄 Viewing shared report: **{metadata.get('dataset_name', 'Report')}**")
        with st.form("shared_viewer_form"):
            st.markdown("**Please introduce yourself to view this report:**")
            v_name    = st.text_input("Your name")
            v_email   = st.text_input("Your email")
            v_company = st.text_input("Company / organization")
            v_role    = st.text_input("Your role")
            submitted = st.form_submit_button("View report")
            if submitted and v_name:
                send_notification_email(
                    event_type="view_link",
                    dataset_name=metadata.get("dataset_name", "Unknown"),
                    format_name="Web link",
                    user_info={"name": v_name, "email": v_email,
                               "company": v_company, "role": v_role},
                    share_url=f"?report={shared_token}",
                )
                st.success("Welcome! Scroll down to see the report.")
                st.markdown(f"### {metadata.get('dataset_name', 'Report')} — Intelligence Report")
                st.markdown(metadata.get("narrative", "Report content not available in this preview."))
                st.info("To see the full interactive report with charts and KPIs, upload the dataset directly.")
    else:
        st.error("Invalid or expired report link.")
    st.stop()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "parquet"])

    groq_key = st.text_input("Groq API key (free, optional)", type="password",
                              placeholder="gsk_...",
                              help="Get free at console.groq.com — enables AI-written report + KPI stories")

    app_url = st.text_input("Your app URL (for share links)",
                             placeholder="https://yourapp.streamlit.app",
                             help="Paste your Streamlit Cloud URL here to generate shareable links")

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
            task_type  = st.selectbox("Task type", ["auto", "classification", "regression"])
            run_btn    = st.button("Launch council", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}"); run_btn = False
    else:
        st.info("Upload a CSV to begin"); run_btn = False

    st.markdown("---")
    st.markdown("### The council")
    for em, name, role in [
        ("🔬", "Ada", "EDA Specialist"), ("⚡", "Max", "ML Engineer"),
        ("💡", "Iris", "Insight Analyst"), ("🔍", "Rex", "Risk Auditor"),
        ("💻", "Cleo", "Code Generator"), ("🧠", "Synthesis", "Report Writer"),
    ]:
        st.markdown(f"{em} **{name}** — {role}")

    st.markdown("---")
    st.markdown("**[⭐ GitHub](https://github.com/YOUR_USERNAME/datamind-ai)**")

# ─── LANDING ────────────────────────────────────────────────────────────────
if not uploaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 5 AI-selected KPIs\nAuto-selected by importance with plain-English story for each one")
    with col2:
        st.markdown("### 📄 PDF + Word export\nProfessional branded reports you can share with your team")
    with col3:
        st.markdown("### 🔗 Share links\nEvery report gets a unique URL — track who opens it")
    if st.button("Try demo (Iris dataset)"):
        from sklearn.datasets import load_iris
        df = load_iris(as_frame=True).frame
        df.to_csv("/tmp/iris_demo.csv", index=False)
        st.success("Demo saved to /tmp/iris_demo.csv — upload it above!")
    st.stop()

with st.expander("Dataset preview"):
    st.dataframe(preview_df, use_container_width=True)

if not run_btn:
    st.info("Configure settings in the sidebar and click **Launch council**.")
    st.stop()

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
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

# ─── RUN PIPELINE ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🤖 Council in session")
agent_container = st.container()
agent_messages  = []

def on_thought(agent_name, emoji, message):
    agent_messages.append((agent_name, emoji, message))
    color_map = {"Ada": "#10b981", "Max": "#3b82f6", "Iris": "#f59e0b",
                 "Rex": "#ef4444", "Cleo": "#8b5cf6", "Synthesis": "#a78bfa"}
    color = color_map.get(agent_name, "#9d9db8")
    with agent_container:
        st.markdown(f"""<div class="agent-card">
          <div class="agent-name" style="color:{color}">{emoji} {agent_name}</div>
          <div class="agent-msg">{message}</div></div>""", unsafe_allow_html=True)

from agents.orchestrator import Orchestrator, AgentMemory
from agents.council import EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent
from agents.synthesis import SynthesisAgent
from src.kpi_engine import select_top_kpis, generate_kpi_stories, generate_data_narrative

memory = AgentMemory()
orch   = Orchestrator(progress_callback=on_thought)
for Cls in [EDAAgent, MLAgent, InsightAgent, CriticAgent, CodeAgent]:
    orch.register_agent(Cls(memory=memory, progress_callback=on_thought))
synth = SynthesisAgent(memory=memory, groq_api_key=groq_key or None,
                       progress_callback=on_thought)
orch.register_agent(synth)

with st.spinner("Council working..."):
    result   = orch.run(df, target_col, task_type)
    findings = result["findings"]

    ada_f   = findings.get("Ada", {})
    max_f   = findings.get("Max", {})
    iris_f  = findings.get("Iris", {})
    rex_f   = findings.get("Rex", {})
    synth_f = findings.get("Synthesis", {})

    # Generate KPIs
    on_thought("Synthesis", "🧠", "Selecting top 5 KPIs from the dataset...")
    kpis = select_top_kpis(df, target_col, max_f.get("feature_importance"), n=5)
    kpis = generate_kpi_stories(kpis, uploaded.name, target_col, task_type,
                                groq_api_key=groq_key or None)

    # Generate narrative
    on_thought("Synthesis", "🧠", "Writing the data story narrative...")
    narrative = generate_data_narrative(kpis, max_f, ada_f, uploaded.name,
                                       target_col, task_type, groq_api_key=groq_key or None)

st.success(f"✅ Council complete in {result['elapsed_seconds']}s!")

# ─── SHARE URL ───────────────────────────────────────────────────────────────
from src.notifications import build_share_url, generate_share_token

base = app_url.strip().rstrip("/") if app_url.strip() else "http://localhost:8501"
share_metadata = {
    "dataset_name": uploaded.name,
    "target_col": target_col,
    "task_type": task_type,
    "narrative": narrative,
    "best_model": max_f.get("best_model", "N/A"),
    "best_score": max_f.get("best_score", 0),
    "generated_at": datetime.now().isoformat(),
}
share_url = build_share_url(base, share_metadata)

st.markdown(f"""
<div class="share-box">
  <p style="color:#a78bfa;font-weight:700;margin:0 0 6px;">🔗 Your shareable report link</p>
  <p style="color:#c5c3e0;font-size:0.85rem;margin:0 0 10px;">Share this with anyone — they'll see the report and you'll get an email with their details.</p>
  <code style="color:#e2e8f0;background:#0d1117;padding:8px 12px;border-radius:6px;
        font-size:0.8rem;display:block;word-break:break-all;">{share_url}</code>
</div>
""", unsafe_allow_html=True)
st.code(share_url, language=None)

# ─── RESULTS TABS ────────────────────────────────────────────────────────────
shape   = ada_f.get("shape", {})
insights = iris_f.get("insights", [])
risks    = rex_f.get("risks", [])
debate   = synth_f.get("debate", [])
leaderboard = max_f.get("leaderboard", [])

tab_kpi, tab_story, tab_models, tab_debate, tab_risks, tab_code, tab_export = st.tabs([
    "📊 KPIs", "📖 Story", "🏆 Models", "🗣️ Debate", "🔍 Risks", "💻 Code", "📥 Export"
])

# ── KPI TAB ──────────────────────────────────────────────────────────────────
with tab_kpi:
    st.markdown("### Top 5 KPIs — AI selected & narrated")
    st.caption("Selected by the council based on predictive power and business relevance.")

    cols = st.columns(2)
    for i, kpi in enumerate(kpis):
        col_name = kpi.get("title", kpi.get("column", "KPI"))
        mean_val = kpi.get("mean", 0)
        trend    = kpi.get("trend_direction", "flat")
        trend_pct = kpi.get("trend_pct", 0)
        corr     = kpi.get("target_correlation")
        story    = kpi.get("story", "")

        if abs(mean_val) >= 1_000_000:
            val_str = f"{mean_val/1_000_000:.2f}M"
        elif abs(mean_val) >= 1_000:
            val_str = f"{mean_val/1_000:.2f}K"
        else:
            val_str = f"{mean_val:.3f}"

        trend_icon  = "▲" if trend == "up" else "▼" if trend == "down" else "●"
        trend_class = f"kpi-trend-{trend}"
        corr_html   = f"<span style='color:#9d9db8;font-size:0.78rem'>vs target r={corr:.2f}</span>" if corr else ""

        with cols[i % 2]:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-lbl">KPI {i+1}</div>
              <div style="font-size:1rem;font-weight:700;color:#e8e6ff;margin:4px 0">{col_name}</div>
              <div class="kpi-val">{val_str}</div>
              <div style="display:flex;gap:12px;margin-top:6px;align-items:center">
                <span class="{trend_class}">{trend_icon} {trend_pct:+.1f}%</span>
                {corr_html}
                <span style="color:#6b6b8a;font-size:0.78rem">n={kpi.get('count',0):,}</span>
              </div>
              <div class="kpi-story">{story}</div>
            </div>""", unsafe_allow_html=True)

# ── STORY TAB ────────────────────────────────────────────────────────────────
with tab_story:
    st.markdown("### Data intelligence narrative")
    if groq_key:
        st.success("Narrative written by Groq LLaMA 3 (free AI)")
    else:
        st.info("Add a free Groq key for AI-written narrative")
    st.markdown(narrative)

# ── MODELS TAB ───────────────────────────────────────────────────────────────
with tab_models:
    st.markdown("### Model leaderboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="metric-val">{shape.get("rows",0):,}</div><div class="metric-lbl">Rows</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="metric-val">{shape.get("cols",0)}</div><div class="metric-lbl">Columns</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="metric-val">{max_f.get("best_score",0):.4f}</div><div class="metric-lbl">Best {max_f.get("metric_name","score")}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">{max_f.get("best_model","N/A")}</div><div class="metric-lbl">Best model</div></div>', unsafe_allow_html=True)

    if leaderboard:
        lb_df = pd.DataFrame([{k: v for k, v in r.items()} for r in leaderboard])
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

    fi = max_f.get("feature_importance")
    if fi:
        st.markdown("#### Top feature importances")
        fi_df = pd.DataFrame({"Feature": fi["features"], "Importance": fi["importances"]})
        st.bar_chart(fi_df.set_index("Feature"), color="#7c3aed")

# ── DEBATE TAB ───────────────────────────────────────────────────────────────
with tab_debate:
    st.markdown("### Agent debate")
    for i, entry in enumerate(debate):
        cls = "debate-left" if i % 2 == 0 else "debate-right"
        st.markdown(f'<div class="debate-bubble {cls}"><strong>{entry["emoji"]} {entry["agent"]}</strong><br>{entry["message"]}</div>', unsafe_allow_html=True)

# ── RISKS TAB ────────────────────────────────────────────────────────────────
with tab_risks:
    st.markdown("### Risk audit (Rex)")
    for r in risks:
        css = f"risk-{'high' if r.get('severity') in ['high','critical'] else r.get('severity','low')}"
        st.markdown(f'<div class="{css}"><strong>[{r.get("severity","").upper()}] {r.get("risk","")}</strong><br>{r.get("detail","")}</div>', unsafe_allow_html=True)
    for ins in insights:
        st.markdown(f'<div class="insight-card">💡 <strong>{ins.get("title","")}</strong><br>{ins.get("detail","")}</div>', unsafe_allow_html=True)

# ── CODE TAB ─────────────────────────────────────────────────────────────────
with tab_code:
    cleo_f = findings.get("Cleo", {})
    for name, code in cleo_f.get("snippets", {}).items():
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.code(code, language="python")

# ── EXPORT TAB ───────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("### Download your report")
    st.markdown("Fill in your details — this helps us understand who's using DataMind AI.")

    with st.form("user_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name    = st.text_input("Your name *", placeholder="Jane Smith")
            u_email   = st.text_input("Your email *", placeholder="jane@company.com")
        with col2:
            u_company = st.text_input("Company / organization", placeholder="Acme Corp")
            u_role    = st.text_input("Your role", placeholder="Data Scientist")

        st.markdown("**Select export format:**")
        fmt_col1, fmt_col2, fmt_col3 = st.columns(3)
        with fmt_col1: want_pdf  = st.checkbox("📄 PDF report", value=True)
        with fmt_col2: want_docx = st.checkbox("📝 Word (.docx)")
        with fmt_col3: want_md   = st.checkbox("📋 Markdown")

        generate_btn = st.form_submit_button("Generate & download", type="primary")

    if generate_btn:
        if not u_name or not u_email:
            st.error("Please enter your name and email to download the report.")
        else:
            user_info = {"name": u_name, "email": u_email,
                         "company": u_company, "role": u_role}

            with st.spinner("Building your reports..."):
                # Common args
                report_args = dict(
                    dataset_name=uploaded.name,
                    target_col=target_col,
                    task_type=task_type,
                    narrative=narrative,
                    kpis=kpis,
                    ml_findings=max_f,
                    eda_findings=ada_f,
                    insights=insights,
                    risks=risks,
                    debate=debate,
                    share_url=share_url,
                )

                dl_col1, dl_col2, dl_col3 = st.columns(3)

                if want_pdf:
                    from src.pdf_report import generate_pdf
                    pdf_bytes = generate_pdf(**report_args)
                    with dl_col1:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"datamind_{uploaded.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            on_click=lambda: None,
                        )

                if want_docx:
                    from src.docx_report import generate_docx
                    docx_bytes = generate_docx(**report_args)
                    with dl_col2:
                        st.download_button(
                            "⬇️ Download Word",
                            data=docx_bytes,
                            file_name=f"datamind_{uploaded.name.split('.')[0]}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )

                if want_md:
                    md_content = f"# DataMind AI — {uploaded.name}\n\n{narrative}\n\n"
                    for kpi in kpis:
                        md_content += f"## KPI: {kpi.get('title','')}\n{kpi.get('story','')}\n\n"
                    with dl_col3:
                        st.download_button(
                            "⬇️ Download Markdown",
                            data=md_content.encode("utf-8"),
                            file_name=f"datamind_{uploaded.name.split('.')[0]}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                # Send email notification
                from src.notifications import send_notification_email
                formats_chosen = ", ".join(
                    [f for f, c in [("PDF", want_pdf), ("Word", want_docx), ("Markdown", want_md)] if c]
                )
                sent = send_notification_email(
                    event_type="download",
                    dataset_name=uploaded.name,
                    format_name=formats_chosen,
                    user_info=user_info,
                    share_url=share_url,
                )
                if sent:
                    st.success("Reports ready! Notification sent.")
                else:
                    st.success("Reports ready! (Configure GMAIL settings in secrets to enable email alerts.)")

    st.markdown("---")
    st.markdown("#### Your share link")
    st.code(share_url, language=None)
    st.caption("Anyone with this link can view a preview of this report. You'll get an email with their details when they open it.")
