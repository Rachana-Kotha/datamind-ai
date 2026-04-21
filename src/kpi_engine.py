"""
DataMind AI — KPI Engine
Auto-selects the 5 most important KPIs from any dataset
and generates a plain-English story for each one.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


GROQ_MODEL = "llama3-70b-8192"


def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq LLaMA 3 with a prompt. Returns text or raises."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.5,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def _compute_kpi_value(df: pd.DataFrame, col: str, target_col: str) -> Dict:
    """Compute a KPI value + trend for a given column."""
    s = df[col].dropna()
    if len(s) == 0:
        return {}

    kpi = {
        "column": col,
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "count": int(len(s)),
        "null_pct": round((df[col].isnull().sum() / len(df)) * 100, 2),
    }

    # Trend: split into two halves and compare
    mid = len(s) // 2
    first_half = s.iloc[:mid].mean()
    second_half = s.iloc[mid:].mean()
    if first_half != 0:
        trend_pct = round((second_half - first_half) / abs(first_half) * 100, 2)
        kpi["trend_pct"] = trend_pct
        kpi["trend_direction"] = "up" if trend_pct > 0 else "down" if trend_pct < 0 else "flat"
    else:
        kpi["trend_pct"] = 0
        kpi["trend_direction"] = "flat"

    # Correlation with target
    try:
        if col != target_col and target_col in df.columns:
            target = df[target_col]
            if pd.api.types.is_numeric_dtype(target):
                corr = float(df[[col, target_col]].corr().iloc[0, 1])
                kpi["target_correlation"] = round(corr, 4)
    except Exception:
        pass

    return kpi


def _score_column_importance(df: pd.DataFrame, col: str, target_col: str,
                              feature_importances: Optional[Dict]) -> float:
    """Score a column for KPI worthiness — higher = more important."""
    score = 0.0
    s = df[col].dropna()

    if len(s) == 0:
        return 0.0

    # Feature importance from ML agent
    if feature_importances and col in feature_importances.get("features", []):
        idx = feature_importances["features"].index(col)
        imp = feature_importances["importances"][idx]
        score += imp * 40

    # Variance (normalized)
    try:
        cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0
        score += min(cv, 2.0) * 5
    except Exception:
        pass

    # Correlation with target
    try:
        if col != target_col and target_col in df.columns:
            target = df[target_col]
            if pd.api.types.is_numeric_dtype(target):
                corr = abs(float(df[[col, target_col]].corr().iloc[0, 1]))
                if not np.isnan(corr):
                    score += corr * 20
    except Exception:
        pass

    # Not too many nulls
    null_pct = df[col].isnull().sum() / len(df)
    score -= null_pct * 10

    # Not an ID column
    if col.lower() in ["id", "uuid", "index", "row"]:
        score -= 50

    # Not constant
    if s.nunique() < 2:
        score -= 50

    return score


def select_top_kpis(
    df: pd.DataFrame,
    target_col: str,
    feature_importances: Optional[Dict] = None,
    n: int = 5
) -> List[Dict]:
    """
    Select the N most important KPIs from the dataset.
    Returns a list of KPI dicts with values, trends, and column metadata.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # Score all numeric columns
    scored = []
    for col in num_cols:
        score = _score_column_importance(df, col, target_col, feature_importances)
        if score > -10:  # filter out junk
            scored.append((col, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_cols = [col for col, _ in scored[:n]]

    # Also include target column as KPI 0 if numeric
    kpis = []
    if pd.api.types.is_numeric_dtype(df[target_col]):
        target_kpi = _compute_kpi_value(df, target_col, target_col)
        target_kpi["is_target"] = True
        target_kpi["rank"] = 0
        kpis.append(target_kpi)

    for i, col in enumerate(top_cols[:n]):
        kpi = _compute_kpi_value(df, col, target_col)
        kpi["is_target"] = False
        kpi["rank"] = i + 1
        kpis.append(kpi)

    return kpis[:n]


def generate_kpi_stories(
    kpis: List[Dict],
    dataset_name: str,
    target_col: str,
    task_type: str,
    groq_api_key: Optional[str] = None,
) -> List[Dict]:
    """
    For each KPI, generate a plain-English story paragraph.
    Uses Groq LLaMA 3 if key provided, otherwise uses smart templates.
    """
    enriched = []
    for kpi in kpis:
        col = kpi.get("column", "Unknown")
        mean = kpi.get("mean", 0)
        trend = kpi.get("trend_direction", "flat")
        trend_pct = kpi.get("trend_pct", 0)
        corr = kpi.get("target_correlation")

        if groq_api_key:
            prompt = f"""You are a business analyst writing a KPI card for a data intelligence report.

Dataset: "{dataset_name}"
KPI metric: "{col}"
Statistics: mean={mean}, median={kpi.get('median', 0)}, std={kpi.get('std', 0)}, min={kpi.get('min', 0)}, max={kpi.get('max', 0)}
Trend: {trend} ({trend_pct:+.1f}% from first half to second half of data)
{"Correlation with target '" + target_col + "': " + str(corr) if corr is not None else ""}

Write a 2-3 sentence story that:
1. Explains what this metric means in plain business English
2. Highlights the most important insight from the numbers
3. Suggests one concrete action based on the data

Be specific, use the actual numbers, and write for a non-technical executive. No jargon."""
            try:
                story = _call_groq(prompt, groq_api_key)
            except Exception:
                story = _template_story(col, mean, trend, trend_pct, corr, target_col)
        else:
            story = _template_story(col, mean, trend, trend_pct, corr, target_col)

        kpi["story"] = story
        kpi["title"] = col.replace("_", " ").title()
        enriched.append(kpi)

    return enriched


def generate_data_narrative(
    kpis: List[Dict],
    ml_findings: Dict,
    eda_findings: Dict,
    dataset_name: str,
    target_col: str,
    task_type: str,
    groq_api_key: Optional[str] = None,
) -> str:
    """Generate the overall data story — the opening narrative of the report."""
    shape = eda_findings.get("shape", {})
    best_model = ml_findings.get("best_model", "the model")
    best_score = ml_findings.get("best_score", 0)
    metric = ml_findings.get("metric_name", "score")
    missing_count = len(eda_findings.get("missing", {}))

    top_kpi = kpis[0] if kpis else {}
    kpi_names = [k.get("title", k.get("column", "")) for k in kpis[:5]]

    if groq_api_key:
        prompt = f"""You are a senior data scientist and storyteller writing the opening narrative of a data intelligence report.

Dataset: "{dataset_name}"
Rows: {shape.get('rows', 0):,} | Columns: {shape.get('cols', 0)}
Task: {task_type} | Target variable: "{target_col}"
Best model: {best_model} ({metric}: {best_score:.4f})
Columns with missing data: {missing_count}
Top KPIs identified: {', '.join(kpi_names)}

Write a compelling 3-4 paragraph data story that:
- Opens with a strong, specific observation about this dataset (not generic)
- Explains what the data reveals about the subject matter
- Highlights the most surprising or important finding
- Ends with what this means for decision-making

Write for a C-suite executive. Be specific, use numbers, and make it engaging. No bullet points."""
        try:
            return _call_groq(prompt, groq_api_key)
        except Exception:
            pass

    # Template fallback
    score_word = "excellent" if best_score > 0.9 else "strong" if best_score > 0.75 else "moderate"
    trend_col = top_kpi.get("title", "the primary metric")
    trend_dir = top_kpi.get("trend_direction", "stable")

    return f"""This analysis examined **{shape.get('rows', 0):,} records** from "{dataset_name}", focusing on predicting **{target_col}** — a {task_type} challenge that touches the core of this dataset's value.

The data tells a clear story: **{trend_col}** shows a {trend_dir} trend, which directly shapes how the predictive model interprets patterns. Across all {shape.get('cols', 0)} variables examined, five KPIs emerged as the most critical signals — not just statistically, but in terms of the business decisions they can inform.

The machine learning analysis tested multiple algorithms and found that **{best_model}** delivered {score_word} performance ({metric}: **{best_score:.4f}**). {"This level of accuracy gives strong confidence in automated decision-making." if best_score > 0.85 else "There is meaningful room for improvement through feature engineering and more data collection."} {"Data quality was a factor — " + str(missing_count) + " columns had missing values, handled automatically through imputation." if missing_count > 0 else "The dataset was clean with no missing values, which contributed to reliable model performance."}

The five KPIs below are not arbitrary — they were selected by the AI council based on their predictive power, business relevance, and the story each one tells about the underlying dynamics of your data."""


def _template_story(col: str, mean: float, trend: str, trend_pct: float,
                    corr: Optional[float], target_col: str) -> str:
    col_name = col.replace("_", " ")
    trend_word = {"up": "increasing", "down": "declining", "flat": "stable"}.get(trend, "stable")
    trend_sentence = f"The metric is **{trend_word}** ({trend_pct:+.1f}% across the dataset), " if trend_pct != 0 else "The metric is stable across the dataset, "

    corr_sentence = ""
    if corr is not None and not np.isnan(corr):
        strength = "strongly" if abs(corr) > 0.7 else "moderately" if abs(corr) > 0.4 else "weakly"
        direction = "positively" if corr > 0 else "negatively"
        corr_sentence = f" It is {strength} {direction} correlated with {target_col.replace('_', ' ')} (r={corr:.2f}), making it a key lever for influencing outcomes."

    return (
        f"**{col_name}** has an average value of **{mean:,.4f}** across the dataset. "
        f"{trend_sentence}suggesting this variable carries meaningful signal over time.{corr_sentence} "
        f"Monitor this metric closely — changes here are likely to precede shifts in your target variable."
    )
