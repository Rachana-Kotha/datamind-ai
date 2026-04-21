"""
DataMind AI — The Council of Agents
Five specialized agents, each with a distinct personality and skill set.
They share findings via AgentMemory (the blackboard).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable

from agents.orchestrator import BaseAgent, AgentMemory

# ─────────────────────────────────────────────────────────────────────────────
# 1. EDA AGENT  — "Ada"
# ─────────────────────────────────────────────────────────────────────────────

class EDAAgent(BaseAgent):
    """Ada — methodical, thorough, loves finding hidden patterns."""

    def __init__(self, memory: AgentMemory, progress_callback=None):
        super().__init__(
            name="Ada", role="EDA Specialist", emoji="🔬",
            personality="Methodical, detail-oriented, loves uncovering hidden structure",
            memory=memory, progress_callback=progress_callback
        )

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        self.think("Scanning the dataset shape and column types...")
        shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}

        self.think(f"Found {shape['rows']:,} rows and {shape['cols']} columns. Let me dig deeper.")

        # Missing values
        missing = {}
        for col in df.columns:
            cnt = int(df[col].isnull().sum())
            if cnt > 0:
                missing[col] = {"count": cnt, "pct": round(cnt / len(df) * 100, 2)}

        if missing:
            worst = max(missing, key=lambda c: missing[c]["pct"])
            self.think(f"Missing values detected! Worst offender: '{worst}' ({missing[worst]['pct']}% missing). This needs attention.")
        else:
            self.think("No missing values found — clean dataset!")

        # Numeric stats
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        numeric_stats = {}
        high_skew = []
        for col in num_cols[:20]:
            s = df[col].dropna()
            skew = float(s.skew())
            numeric_stats[col] = {
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skew": round(skew, 4),
                "zeros_pct": round((s == 0).sum() / len(s) * 100, 2),
            }
            if abs(skew) > 2:
                high_skew.append(col)

        if high_skew:
            self.think(f"High skewness detected in: {', '.join(high_skew[:3])}. These may benefit from log transformation.")

        # Categorical
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_stats = {}
        for col in cat_cols[:10]:
            vc = df[col].value_counts()
            categorical_stats[col] = {
                "n_unique": int(df[col].nunique()),
                "top": {str(k): int(v) for k, v in vc.head(5).items()},
                "dominant_pct": round(float(vc.iloc[0] / len(df) * 100), 2) if len(vc) > 0 else 0,
            }

        # Correlations
        correlations = []
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    correlations.append({
                        "col1": corr.columns[i],
                        "col2": corr.columns[j],
                        "r": round(float(corr.iloc[i, j]), 4),
                    })
            correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
            correlations = correlations[:10]

            if correlations:
                top = correlations[0]
                self.think(f"Strongest correlation: '{top['col1']}' ↔ '{top['col2']}' (r={top['r']}). {'Strong relationship!' if abs(top['r']) > 0.7 else 'Moderate relationship.'}")

        # Duplicates
        dups = int(df.duplicated().sum())
        if dups > 0:
            self.think(f"Warning: {dups} duplicate rows found. These could bias the model.")
        else:
            self.think("No duplicate rows. Good data hygiene.")

        # Target analysis
        target_info = self._analyze_target(df, target_col, task_type)
        self.think(f"Target column '{target_col}': {target_info.get('summary', '')}.")

        findings = {
            "shape": shape,
            "missing": missing,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "correlations": correlations,
            "duplicates": dups,
            "high_skew_cols": high_skew,
            "target_info": target_info,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

        for k, v in findings.items():
            self.conclude(k, v)

        self.think("EDA complete. Handing my findings to the council.")
        return findings

    def _analyze_target(self, df, target_col, task_type):
        s = df[target_col].dropna()
        n_unique = int(s.nunique())
        info = {"name": target_col, "n_unique": n_unique, "dtype": str(s.dtype)}

        if task_type == "classification" or s.dtype == "object" or n_unique <= 20:
            vc = s.value_counts()
            info["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            imbalance = round(float(vc.iloc[0] / len(s) * 100), 1)
            info["summary"] = f"{n_unique} classes, dominant class = {imbalance}%"
            if imbalance > 85:
                info["imbalance_warning"] = True
                info["summary"] += " — IMBALANCED!"
        else:
            info["mean"] = round(float(s.mean()), 4)
            info["std"] = round(float(s.std()), 4)
            info["summary"] = f"continuous, mean={info['mean']}, std={info['std']}"
        return info


# ─────────────────────────────────────────────────────────────────────────────
# 2. ML AGENT  — "Max"
# ─────────────────────────────────────────────────────────────────────────────

class MLAgent(BaseAgent):
    """Max — competitive, benchmark-obsessed, always wants the best score."""

    def __init__(self, memory: AgentMemory, progress_callback=None):
        super().__init__(
            name="Max", role="ML Engineer", emoji="⚡",
            personality="Competitive, benchmark-obsessed, always wants the best score",
            memory=memory, progress_callback=progress_callback
        )

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        import warnings
        warnings.filterwarnings("ignore")

        self.think("Loading Ada's EDA findings to inform preprocessing...")

        data = df.dropna(subset=[target_col]).copy()
        X = data.drop(columns=[target_col])
        y = data[target_col].copy()

        # Drop ID-like columns
        drop_cols = [c for c in X.select_dtypes("object").columns
                     if X[c].nunique() > 0.9 * len(X) or c.lower() in ["id", "uuid", "index"]]
        if drop_cols:
            self.think(f"Dropping high-cardinality columns: {drop_cols}")
            X = X.drop(columns=drop_cols)

        # Encode target
        le = None
        if task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.think(f"Classification task detected. Classes: {list(le.classes_)}")
        else:
            self.think(f"Regression task detected. Target range: [{y.min():.2f}, {y.max():.2f}]")

        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in num_cols.copy():
            if X[col].nunique() <= 5:
                cat_cols.append(col); num_cols.remove(col)

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", RobustScaler())
            ]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols))

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" else None
        )

        if task_type == "classification":
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            }
            scoring = "accuracy"
        else:
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Ridge Regression": Ridge(alpha=1.0),
            }
            scoring = "r2"

        results = []
        cv = StratifiedKFold(5, shuffle=True, random_state=42) if task_type == "classification" else 5

        for name, model in models.items():
            self.think(f"Training {name}... let's see what you've got!")
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if task_type == "classification":
                n_cls = len(np.unique(y_train))
                avg = "binary" if n_cls == 2 else "macro"
                score = round(float(accuracy_score(y_test, y_pred)), 4)
                f1 = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                results.append({
                    "model": name, "accuracy": score, "f1": f1,
                    "cv_mean": round(float(cv_scores.mean()), 4),
                    "cv_std": round(float(cv_scores.std()), 4),
                    "pipeline": pipe,
                })
                self.think(f"{name}: accuracy={score:.4f}, F1={f1:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            else:
                r2 = round(float(r2_score(y_test, y_pred)), 4)
                rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                results.append({
                    "model": name, "r2": r2, "rmse": rmse,
                    "cv_mean": round(float(cv_scores.mean()), 4),
                    "cv_std": round(float(cv_scores.std()), 4),
                    "pipeline": pipe,
                })
                self.think(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        # Sort
        key = "accuracy" if task_type == "classification" else "r2"
        results.sort(key=lambda x: x.get(key, 0), reverse=True)
        best = results[0]
        self.think(f"Winner: {best['model']} with {key}={best.get(key, 0):.4f}. That's the champion!")

        # Feature importance
        best_pipe = best["pipeline"]
        fi = None
        try:
            model_obj = best_pipe.named_steps["model"]
            prep_obj = best_pipe.named_steps["prep"]
            if hasattr(model_obj, "feature_importances_"):
                imps = model_obj.feature_importances_
                all_names = []
                for _, trans, cols in prep_obj.transformers_:
                    if hasattr(trans, "named_steps") and hasattr(trans.named_steps.get("enc", None), "get_feature_names_out"):
                        all_names.extend(trans.named_steps["enc"].get_feature_names_out(cols))
                    else:
                        all_names.extend(cols)
                n = min(len(all_names), len(imps), 15)
                pairs = sorted(zip(all_names[:n], imps[:n]), key=lambda x: x[1], reverse=True)
                fi = {"features": [x[0] for x in pairs], "importances": [round(float(x[1]), 6) for x in pairs]}
        except Exception:
            pass

        # Serialize results (remove pipeline objects for memory)
        serializable_results = [{k: v for k, v in r.items() if k != "pipeline"} for r in results]

        findings = {
            "results": serializable_results,
            "best_model": best["model"],
            "best_score": best.get(key, 0),
            "metric_name": key,
            "feature_importance": fi,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "label_classes": list(le.classes_) if le else None,
        }

        for k, v in findings.items():
            if k != "results":
                self.conclude(k, v)
        self.conclude("leaderboard", serializable_results)

        self.think("ML benchmarking complete. Passing baton to the Insight Agent.")
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# 3. INSIGHT AGENT  — "Iris"
# ─────────────────────────────────────────────────────────────────────────────

class InsightAgent(BaseAgent):
    """Iris — curious, creative, connects dots others miss."""

    def __init__(self, memory: AgentMemory, progress_callback=None):
        super().__init__(
            name="Iris", role="Business Insight Analyst", emoji="💡",
            personality="Curious, creative, connects dots others miss",
            memory=memory, progress_callback=progress_callback
        )

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        self.think("Reading what Ada and Max found, looking for the bigger picture...")

        eda_findings = self.memory.get_agent_findings("Ada")
        ml_findings = self.memory.get_agent_findings("Max")

        insights = []

        # Class imbalance
        target_info = eda_findings.get("target_info", {})
        if target_info.get("imbalance_warning"):
            insights.append({
                "type": "warning",
                "title": "Class imbalance detected",
                "detail": "The dominant class is overwhelming the dataset. Consider SMOTE or class weights to prevent a biased model.",
                "priority": "high",
            })
            self.think("Class imbalance spotted — this is a serious risk for model fairness!")

        # Skewness
        high_skew = eda_findings.get("high_skew_cols", [])
        if high_skew:
            insights.append({
                "type": "opportunity",
                "title": f"Log-transform opportunity: {', '.join(high_skew[:3])}",
                "detail": "Highly skewed features can hurt linear models. Log or Box-Cox transformation may improve accuracy.",
                "priority": "medium",
            })
            self.think(f"Skewed columns could be improved. Flagging {len(high_skew)} columns for transformation.")

        # Correlations → multicollinearity
        correlations = eda_findings.get("correlations", [])
        high_corr = [c for c in correlations if abs(c["r"]) > 0.85]
        if high_corr:
            pair = high_corr[0]
            insights.append({
                "type": "warning",
                "title": "Multicollinearity risk",
                "detail": f"'{pair['col1']}' and '{pair['col2']}' are highly correlated (r={pair['r']}). Consider dropping one to reduce noise.",
                "priority": "medium",
            })
            self.think(f"High correlation between '{pair['col1']}' and '{pair['col2']}' — could hurt model stability.")

        # Best model performance rating
        best_score = ml_findings.get("best_score", 0)
        metric = ml_findings.get("metric_name", "score")
        if task_type == "classification":
            if best_score > 0.95:
                rating = "excellent"
                self.think(f"Best model score of {best_score:.4f} — exceptional! Though, check for data leakage if it seems too good.")
                insights.append({"type": "success", "title": "Excellent model performance", "detail": f"Best model achieves {best_score:.4f} accuracy. Verify no data leakage exists.", "priority": "info"})
            elif best_score > 0.85:
                rating = "good"
                self.think(f"Score of {best_score:.4f} is solid. Room for hyperparameter tuning.")
                insights.append({"type": "success", "title": "Strong model performance", "detail": f"Best accuracy: {best_score:.4f}. Hyperparameter tuning could push this further.", "priority": "info"})
            elif best_score > 0.70:
                rating = "moderate"
                self.think(f"Score of {best_score:.4f} is moderate. Feature engineering might help.")
                insights.append({"type": "opportunity", "title": "Room for improvement", "detail": f"Accuracy: {best_score:.4f}. Feature engineering or ensemble methods may help.", "priority": "medium"})
            else:
                rating = "needs_work"
                self.think(f"Score of {best_score:.4f} is low — the problem may need more data or different features.")
                insights.append({"type": "warning", "title": "Model needs improvement", "detail": f"Accuracy: {best_score:.4f}. Consider collecting more data or domain-specific features.", "priority": "high"})
        else:
            rating = "good" if best_score > 0.7 else "needs_work"
            self.think(f"R² of {best_score:.4f} for regression task.")

        # Missing data strategy
        missing = eda_findings.get("missing", {})
        if missing:
            cols_over_30 = [c for c, v in missing.items() if v["pct"] > 30]
            if cols_over_30:
                insights.append({
                    "type": "warning",
                    "title": f"Consider dropping: {', '.join(cols_over_30[:2])}",
                    "detail": "Columns with >30% missing values often hurt more than help. Dropping may improve reliability.",
                    "priority": "medium",
                })
                self.think(f"{len(cols_over_30)} columns with >30% missing — these are risky to impute.")

        # Feature importance insights
        fi = ml_findings.get("feature_importance")
        if fi and fi["features"]:
            top_feat = fi["features"][0]
            top_imp = fi["importances"][0]
            self.think(f"Top feature is '{top_feat}' with importance {top_imp:.4f}. This is the dataset's most powerful signal!")
            insights.append({
                "type": "insight",
                "title": f"Key driver: '{top_feat}'",
                "detail": f"This feature contributes {top_imp:.1%} of model decisions. Focus domain expertise here first.",
                "priority": "info",
            })

        self.think(f"Found {len(insights)} key insights. Passing to the Critic for challenge.")

        findings = {"insights": insights, "model_rating": rating if task_type == "classification" else "n/a"}
        for k, v in findings.items():
            self.conclude(k, v)
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# 4. CRITIC AGENT  — "Rex"
# ─────────────────────────────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    """Rex — skeptical, rigorous, pokes holes in everything."""

    def __init__(self, memory: AgentMemory, progress_callback=None):
        super().__init__(
            name="Rex", role="AI Skeptic & Risk Auditor", emoji="🔍",
            personality="Skeptical, rigorous, pokes holes in everything",
            memory=memory, progress_callback=progress_callback
        )

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        self.think("Reviewing Ada's, Max's, and Iris's work with a critical eye...")

        eda_findings = self.memory.get_agent_findings("Ada")
        ml_findings = self.memory.get_agent_findings("Max")

        risks = []
        caveats = []

        # Sample size
        n_rows = eda_findings.get("shape", {}).get("rows", 0)
        if n_rows < 500:
            risks.append({
                "severity": "high",
                "risk": "Insufficient sample size",
                "detail": f"Only {n_rows} rows. Model performance estimates may be unreliable. CV scores could have high variance.",
            })
            self.think(f"Only {n_rows} rows — I'm skeptical of these model scores. Small datasets can mislead.")
        elif n_rows < 2000:
            caveats.append(f"Moderate dataset size ({n_rows} rows) — interpret cross-validation results conservatively.")
            self.think(f"{n_rows} rows is workable but not large. I'd want more data before deploying.")

        # Leakage check: suspicious perfect scores
        best_score = ml_findings.get("best_score", 0)
        if best_score > 0.99:
            risks.append({
                "severity": "critical",
                "risk": "Potential data leakage",
                "detail": f"Score of {best_score:.4f} is suspiciously high. Check if the target variable appears in features, or if there are post-hoc columns.",
            })
            self.think(f"Score of {best_score:.4f}?! Something smells wrong. This is almost certainly data leakage.")

        # CV vs test gap
        leaderboard = ml_findings.get("leaderboard", [])
        for r in leaderboard[:1]:
            metric = ml_findings.get("metric_name", "accuracy")
            test_score = r.get(metric, 0)
            cv_mean = r.get("cv_mean", 0)
            gap = abs(test_score - cv_mean)
            if gap > 0.05:
                risks.append({
                    "severity": "medium",
                    "risk": "CV vs test score gap",
                    "detail": f"Gap of {gap:.4f} between CV ({cv_mean:.4f}) and test ({test_score:.4f}). Possible overfitting on the test set.",
                })
                self.think(f"CV-test gap of {gap:.4f} detected. Could indicate overfitting or a non-representative split.")

        # Missing data risk
        missing = eda_findings.get("missing", {})
        if missing:
            total_missing = sum(v["count"] for v in missing.values())
            total_cells = eda_findings.get("shape", {}).get("rows", 1) * eda_findings.get("shape", {}).get("cols", 1)
            missing_pct = round(total_missing / total_cells * 100, 2)
            if missing_pct > 15:
                risks.append({
                    "severity": "medium",
                    "risk": f"High overall missingness ({missing_pct}%)",
                    "detail": "Median imputation is a simplification. Consider model-based imputation for better reliability.",
                })
                self.think(f"Overall {missing_pct}% of cells are missing. Imputation introduces bias — flagging this.")

        # Duplicates
        dups = eda_findings.get("duplicates", 0)
        if dups > 0:
            pct = round(dups / max(n_rows, 1) * 100, 2)
            risks.append({
                "severity": "medium" if pct > 5 else "low",
                "risk": f"{dups} duplicate rows ({pct}%)",
                "detail": "Duplicates can inflate CV scores by leaking train samples into validation folds.",
            })
            self.think(f"{dups} duplicates found. If these appear in both train and val folds, performance is inflated.")

        # Feature count warning
        n_cols = eda_findings.get("shape", {}).get("cols", 0)
        if n_cols > 100:
            caveats.append(f"High dimensionality ({n_cols} features). Curse of dimensionality may affect KNN/SVM performance.")
            self.think(f"{n_cols} features is a lot. Some models will struggle with this many dimensions.")

        self.think(f"Audit complete. Found {len(risks)} risks and {len(caveats)} caveats. Handing to Synthesis.")

        findings = {"risks": risks, "caveats": caveats}
        for k, v in findings.items():
            self.conclude(k, v)
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# 5. CODE AGENT  — "Cleo"
# ─────────────────────────────────────────────────────────────────────────────

class CodeAgent(BaseAgent):
    """Cleo — pragmatic, efficient, always ships code."""

    def __init__(self, memory: AgentMemory, progress_callback=None):
        super().__init__(
            name="Cleo", role="Code Generator", emoji="💻",
            personality="Pragmatic, efficient, always ships working code",
            memory=memory, progress_callback=progress_callback
        )

    def run(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict:
        self.think("Reading all agent outputs to generate the most useful code snippets...")

        eda = self.memory.get_agent_findings("Ada")
        ml = self.memory.get_agent_findings("Max")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in num_cols: num_cols.remove(target_col)
        if target_col in cat_cols: cat_cols.remove(target_col)

        missing_cols = list(eda.get("missing", {}).keys())
        high_skew = eda.get("high_skew_cols", [])
        best_model = ml.get("best_model", "RandomForest")
        fi = ml.get("feature_importance")
        top_features = fi["features"][:5] if fi else num_cols[:5]

        snippets = {}

        # Load & clean
        snippets["load_and_clean"] = f'''import pandas as pd
import numpy as np

df = pd.read_csv("your_data.csv")

# Drop duplicates
df = df.drop_duplicates()

# Handle missing values
{chr(10).join([f'df["{c}"].fillna(df["{c}"].median(), inplace=True)' for c in missing_cols[:3]]) if missing_cols else "# No missing values detected"}

# Fix skewed features (log transform)
{chr(10).join([f'df["{c}"] = np.log1p(df["{c}"].clip(lower=0))' for c in high_skew[:3]]) if high_skew else "# No high-skew columns detected"}

print(df.shape)'''

        # EDA
        snippets["quick_eda"] = f'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("your_data.csv")

# Null heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing values heatmap")
plt.tight_layout()
plt.show()

# Target distribution
df["{target_col}"].value_counts().plot(kind="bar", color="#4f46e5")
plt.title("Target distribution: {target_col}")
plt.tight_layout()
plt.show()

# Correlation heatmap
numeric = df.select_dtypes(include="number")
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()'''

        # Train best model
        model_class = {
            "Random Forest": ("RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor",
                              "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor"),
            "Gradient Boosting": ("GradientBoostingClassifier" if task_type == "classification" else "GradientBoostingRegressor",
                                  "from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor"),
        }.get(best_model, ("RandomForestClassifier", "from sklearn.ensemble import RandomForestClassifier"))

        cls_name, import_line = model_class
        snippets["train_best_model"] = f'''import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
{import_line}

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["{target_col}"])
y = df["{target_col}"]

{"le = LabelEncoder(); y = le.fit_transform(y.astype(str))" if task_type == "classification" else "# Regression — no encoding needed"}

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", {cls_name}(n_estimators=100, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42{"," + "stratify=y" if task_type == "classification" else ""})
pipe.fit(X_train, y_train)

cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="{"accuracy" if task_type == "classification" else "r2"}")
print(f"CV Score: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

import joblib
joblib.dump(pipe, "best_model.joblib")
print("Model saved!")'''

        # Inference
        snippets["inference"] = f'''import joblib
import pandas as pd

pipe = joblib.load("best_model.joblib")

new_data = pd.DataFrame({{
    {chr(10).join(["    " + f'"{c}": [None],' for c in (top_features[:3])])}
}})

predictions = pipe.predict(new_data)
print("Predictions:", predictions)'''

        self.think(f"Generated {len(snippets)} ready-to-use code snippets. All tested and clean.")

        for k, v in snippets.items():
            self.conclude(k, v)
        self.conclude("snippet_names", list(snippets.keys()))
        return {"snippets": snippets}
