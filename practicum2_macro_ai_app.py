from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Practicum II Predictive Analysis on Geopolitical Risk and Macro Ec", layout="wide")


@dataclass
class LoadedTable:
    name: str
    df: pd.DataFrame | None
    source: str


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = APP_DIR
DEFAULT_PROCESSED_DIR = Path(
    os.getenv(
        "PRACTICUM2_PROCESSED_DIR",
        str(DEFAULT_DATA_ROOT / "Processed Data"),
    )
)

TABLE_SPECS = {
    "step5_results": {
        "filename": "clean_step5_results.csv",
        "required": ["target", "window", "model", "RMSE", "R2"],
    },
    "step5_incremental": {
        "filename": "clean_step5_incremental_gpr.csv",
        "required": ["target", "window", "model", "delta_RMSE_vs_LagLinear", "delta_R2_vs_LagLinear"],
    },
    "step7_links": {
        "filename": "clean_step7_links.csv",
        "required": ["best_lag_months", "best_corr", "abs_corr"],
    },
    "step7_event": {
        "filename": "clean_step7_event.csv",
        "required": ["event", "category", "delta_post_minus_pre"],
    },
    "step7_summary": {
        "filename": "clean_step7_summary.csv",
        "required": ["category", "mean_signed_delta", "mean_abs_delta"],
    },
    "step8_final": {
        "filename": "clean_step8_final_summary.csv",
        "required": ["target", "model", "RMSE", "R2"],
    },
}


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_csv_from_bytes(payload: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(payload))


def autoload_table(name: str, processed_dir: Path) -> LoadedTable:
    spec = TABLE_SPECS[name]
    target_path = processed_dir / spec["filename"]
    if target_path.exists():
        df = _safe_read_csv(target_path)
        if df is not None:
            return LoadedTable(name=name, df=df, source=str(target_path))
    return LoadedTable(name=name, df=None, source="not found")


def _candidate_processed_dirs() -> list[Path]:
    candidates = [
        DEFAULT_PROCESSED_DIR,
        APP_DIR / "Processed Data",
        APP_DIR,
    ]
    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def pick_best_processed_dir() -> Path:
    best_dir = _candidate_processed_dirs()[0]
    best_count = -1
    for d in _candidate_processed_dirs():
        count = 0
        for spec in TABLE_SPECS.values():
            if (d / spec["filename"]).exists():
                count += 1
        if count > best_count:
            best_count = count
            best_dir = d
    return best_dir


def validate_table(name: str, df: pd.DataFrame) -> list[str]:
    required = TABLE_SPECS[name]["required"]
    return [col for col in required if col not in df.columns]


def score_step5(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    if "window" not in scored.columns:
        scored["window"] = "unknown"
    scored["best_rmse"] = scored.groupby(["target", "window"])["RMSE"].transform("min")
    scored["performance_pct"] = (scored["best_rmse"] / scored["RMSE"]) * 100.0
    return scored


def build_rule_based_brief(data: dict[str, pd.DataFrame]) -> str:
    lines: list[str] = []

    step5 = data.get("step5_results")
    if isinstance(step5, pd.DataFrame) and len(step5):
        s5 = step5.copy()
        if "window" in s5.columns and (s5["window"] == "full_test").any():
            s5 = s5[s5["window"] == "full_test"].copy()
        winners = (
            s5.sort_values(["target", "RMSE", "R2"], ascending=[True, True, False])
            .groupby("target", as_index=False)
            .head(1)
        )
        top_models = winners["model"].value_counts().to_dict()
        lines.append(f"Step 5: best models by target -> {top_models}.")

    step7 = data.get("step7_summary")
    if isinstance(step7, pd.DataFrame) and len(step7) and "mean_abs_delta" in step7.columns:
        top_row = step7.sort_values("mean_abs_delta", ascending=False).iloc[0]
        lines.append(
            f"Step 7: most event-sensitive category is {top_row.get('category', 'N/A')} "
            f"(avg abs change {float(top_row.get('mean_abs_delta', np.nan)):.3f})."
        )

    step8 = data.get("step8_final")
    if isinstance(step8, pd.DataFrame) and len(step8):
        best_overall = step8.sort_values("RMSE", ascending=True).iloc[0]
        lines.append(
            f"Step 8: strongest target-level fit is {best_overall.get('target', 'N/A')} "
            f"with {best_overall.get('model', 'N/A')} (RMSE={float(best_overall.get('RMSE', np.nan)):.3f})."
        )

    if not lines:
        return "Upload Step 5/7/8 outputs to generate an AI-style project brief."

    lines.append("Interpretation: short-horizon lag structure appears dominant, while category responses around shocks are heterogeneous.")
    return "\n".join(lines)


def query_ollama(model_name: str, prompt: str, base_url: str = "http://localhost:11434") -> str:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def ai_prompt(question: str, data: dict[str, pd.DataFrame]) -> str:
    snippets: list[str] = []
    for key, df in data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        sample = df.head(8).to_dict(orient="records")
        snippets.append(f"[{key}] columns={list(df.columns)} sample={sample}")

    context = "\n".join(snippets) if snippets else "No data loaded."
    return (
        "You are a macroeconomics analysis assistant for Practicum II. "
        "Use only the provided table snippets and avoid inventing values. "
        "Give concise, plain-English findings.\n\n"
        f"Data context:\n{context}\n\n"
        f"User question: {question}\n"
        "Answer with: 1) key finding, 2) evidence from data, 3) caveat."
    )


def fallback_answer(question: str, data: dict[str, pd.DataFrame]) -> str:
    q = question.lower()

    lines: list[str] = []

    step5 = data.get("step5_results")
    if isinstance(step5, pd.DataFrame) and len(step5):
        s5 = step5.copy()
        if "window" in s5.columns and (s5["window"] == "full_test").any():
            s5 = s5[s5["window"] == "full_test"].copy()
        winners = (
            s5.sort_values(["target", "RMSE", "R2"], ascending=[True, True, False])
            .groupby("target", as_index=False)
            .head(1)
        )
        if len(winners):
            mode_model = str(winners["model"].mode().iloc[0])
            lines.append(f"Key finding: the most consistent best model is {mode_model}.")
            lines.append(
                "Evidence: this model appears most often among best-by-target results in Step 5 full_test rows."
            )

        if "hardest" in q or "difficult" in q:
            hardest = (
                s5.sort_values(["RMSE"], ascending=False)
                .iloc[0]
                if len(s5)
                else None
            )
            if hardest is not None:
                lines.append(
                    f"Extra: the hardest target in this view is {hardest.get('target', 'N/A')} "
                    f"(RMSE={float(hardest.get('RMSE', np.nan)):.3f})."
                )

    step7 = data.get("step7_summary")
    if isinstance(step7, pd.DataFrame) and len(step7) and "mean_abs_delta" in step7.columns:
        top = step7.sort_values("mean_abs_delta", ascending=False).iloc[0]
        lines.append(
            f"Category signal: {top.get('category', 'N/A')} is most event-sensitive "
            f"(mean_abs_delta={float(top.get('mean_abs_delta', np.nan)):.3f})."
        )

    if not lines:
        return (
            "I could not run Ollama and there is not enough loaded data to infer a safe answer. "
            "Please load Step 5/7/8 CSV outputs first."
        )

    lines.append(
        "Caveat: this fallback answer is rule-based from loaded tables; for richer narrative, run Ollama locally."
    )
    return "\n".join(lines)


def render_header() -> None:
    st.title("Practicum II Predictive Analysis on Geopolitical Risk and Macro Ec")
    st.caption("Interactive interface for Steps 5-8 outputs with optional local AI analyst support.")


def render_sidebar() -> tuple[Path, dict[str, LoadedTable]]:
    st.sidebar.header("Data Loading")
    processed_dir = pick_best_processed_dir()
    st.sidebar.success("Auto-load enabled")
    st.sidebar.caption("CSV files are loaded automatically from the best detected folder.")
    st.sidebar.write(f"Folder: {processed_dir}")

    loaded: dict[str, LoadedTable] = {}
    for name in TABLE_SPECS:
        loaded[name] = autoload_table(name, processed_dir)

    return processed_dir, loaded


def render_data_health(loaded: dict[str, LoadedTable]) -> None:
    st.subheader("Data Health")
    rows: list[dict[str, Any]] = []
    for name, obj in loaded.items():
        ok = isinstance(obj.df, pd.DataFrame)
        if ok:
            missing = validate_table(name, obj.df)
            status = "ok" if not missing else f"missing columns: {missing}"
            rows.append(
                {
                    "table": TABLE_SPECS[name]["filename"],
                    "status": status,
                    "rows": len(obj.df),
                    "source": obj.source,
                }
            )
        else:
            rows.append(
                {
                    "table": TABLE_SPECS[name]["filename"],
                    "status": "not loaded",
                    "rows": 0,
                    "source": obj.source,
                }
            )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_step5(loaded: dict[str, LoadedTable]) -> None:
    st.subheader("Models")
    obj = loaded["step5_results"]
    if obj.df is None:
        st.warning("Step 5 results are not loaded.")
        return

    missing = validate_table("step5_results", obj.df)
    if missing:
        st.error(f"Step 5 data missing columns: {missing}")
        return

    df = obj.df.copy()
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df = df.dropna(subset=["RMSE", "R2"])

    window_options = sorted(df["window"].astype(str).unique())
    selected_window = st.selectbox("Window", window_options, index=0)
    view = df[df["window"].astype(str) == selected_window].copy()

    if view.empty:
        st.info("No rows for the selected window.")
        return

    winners = (
        view.sort_values(["target", "RMSE", "R2"], ascending=[True, True, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    c1, c2 = st.columns(2)
    c1.metric("Targets evaluated", f"{view['target'].nunique()}")
    c2.metric("Most frequent best model", str(winners["model"].mode().iloc[0]))

    fig_rmse = px.bar(
        view,
        x="target",
        y="RMSE",
        color="model",
        barmode="group",
        title=f"RMSE by Model ({selected_window})",
    )
    fig_rmse.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_rmse, use_container_width=True)

    scored = score_step5(view)
    fig_score = px.bar(
        scored,
        x="target",
        y="performance_pct",
        color="model",
        barmode="group",
        title="Relative Performance Score (Best per target = 100)",
    )
    fig_score.add_hline(y=100, line_dash="dot", line_color="#7F8C8D")
    fig_score.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_score, use_container_width=True)

    st.markdown("Best model per target")
    st.dataframe(winners[["target", "model", "RMSE", "R2"]], use_container_width=True)

    inc_obj = loaded["step5_incremental"]
    if inc_obj.df is not None:
        inc = inc_obj.df.copy()
        if {"delta_RMSE_vs_LagLinear", "model", "target"}.issubset(inc.columns):
            inc["delta_RMSE_vs_LagLinear"] = pd.to_numeric(inc["delta_RMSE_vs_LagLinear"], errors="coerce")
            fig_inc = px.bar(
                inc,
                x="target",
                y="delta_RMSE_vs_LagLinear",
                color="model",
                barmode="group",
                title="Incremental RMSE vs LagLinear (negative is better)",
            )
            fig_inc.add_hline(y=0, line_dash="dot", line_color="#7F8C8D")
            fig_inc.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_inc, use_container_width=True)


def render_step7(loaded: dict[str, LoadedTable]) -> None:
    st.subheader("Categories")

    links_obj = loaded["step7_links"]
    event_obj = loaded["step7_event"]
    summary_obj = loaded["step7_summary"]

    if links_obj.df is not None and {"abs_corr"}.issubset(links_obj.df.columns):
        links = links_obj.df.copy()
        x_col = "category" if "category" in links.columns else "channel"
        if x_col not in links.columns:
            st.warning("Step 7 links table is loaded but missing both 'category' and 'channel' columns.")
            links = pd.DataFrame()
        links["abs_corr"] = pd.to_numeric(links["abs_corr"], errors="coerce")
        links = links.dropna(subset=["abs_corr"])  # keep only valid numeric values
        if not links.empty:
            fig_links = px.bar(
                links.sort_values("abs_corr", ascending=False),
                x=x_col,
                y="abs_corr",
                color="source" if "source" in links.columns else None,
                text="abs_corr",
                title="Lead-Lag Transmission Strength",
            )
            fig_links.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_links.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_links, use_container_width=True)

    if event_obj.df is not None and {"event", "category", "delta_post_minus_pre"}.issubset(event_obj.df.columns):
        event = event_obj.df.copy()
        event["delta_post_minus_pre"] = pd.to_numeric(event["delta_post_minus_pre"], errors="coerce")
        event = event.dropna(subset=["delta_post_minus_pre"])
        fig_event = px.bar(
            event,
            x="event",
            y="delta_post_minus_pre",
            color="category",
            barmode="group",
            title="Category Change Around Events (Post 6m - Pre 6m)",
        )
        fig_event.add_hline(y=0, line_dash="dot", line_color="#7F8C8D")
        fig_event.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_event, use_container_width=True)

    if summary_obj.df is not None and {"category", "mean_abs_delta"}.issubset(summary_obj.df.columns):
        s7 = summary_obj.df.copy()
        s7["mean_abs_delta"] = pd.to_numeric(s7["mean_abs_delta"], errors="coerce")
        s7 = s7.dropna(subset=["mean_abs_delta"])
        st.markdown("Average absolute category sensitivity")
        st.dataframe(s7.sort_values("mean_abs_delta", ascending=False), use_container_width=True)


def render_final(loaded: dict[str, LoadedTable]) -> None:
    st.subheader("Final")
    obj = loaded["step8_final"]
    if obj.df is None:
        st.warning("Final summary table is not loaded.")
        return

    missing = validate_table("step8_final", obj.df)
    if missing:
        st.error(f"Step 8 data missing columns: {missing}")
        return

    final_df = obj.df.copy()
    final_df["RMSE"] = pd.to_numeric(final_df["RMSE"], errors="coerce")
    final_df["R2"] = pd.to_numeric(final_df["R2"], errors="coerce")

    st.dataframe(final_df, use_container_width=True)

    fig_final = px.bar(
        final_df,
        x="target",
        y="RMSE",
        color="model",
        title="Best Selected Models by Target (Final Output)",
    )
    fig_final.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_final, use_container_width=True)


def render_ai_agent(loaded: dict[str, LoadedTable]) -> None:
    st.subheader("AI Analyst Agent")
    st.caption("Rule-based briefing is always available. Optional local LLM via Ollama can answer custom questions.")

    tables = {
        key: val.df
        for key, val in loaded.items()
        if isinstance(val.df, pd.DataFrame)
    }

    with st.expander("Auto-brief from loaded data", expanded=True):
        brief = build_rule_based_brief(tables)
        st.text_area("Agent brief", value=brief, height=180)

    st.markdown("Ask a custom question")
    question = st.text_input(
        "Example: Which target is hardest to predict and what does that imply?",
        value="Which model pattern is most consistent across targets?",
    )

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.text_input("Ollama model", value="llama3.1:8b")
    with c2:
        base_url = st.text_input("Ollama URL", value="http://localhost:11434")

    if st.button("Run AI analysis", type="primary"):
        if not question.strip():
            st.warning("Please provide a question.")
            return

        prompt = ai_prompt(question, tables)
        with st.spinner("Running local AI analysis..."):
            try:
                answer = query_ollama(model_name=model_name.strip(), prompt=prompt, base_url=base_url.strip())
                if answer:
                    st.success("AI response generated.")
                    st.write(answer)
                else:
                    st.warning("Model returned an empty response.")
            except URLError:
                st.warning("Could not reach Ollama server. Showing fallback local analysis instead.")
                st.write(fallback_answer(question, tables))
            except Exception as exc:
                st.warning(f"AI analysis via Ollama failed: {exc}")
                st.write(fallback_answer(question, tables))

    st.code(
        "ollama serve\n"
        "ollama pull llama3.1:8b\n"
        "streamlit run practicum2_macro_ai_app.py",
        language="bash",
    )


def main() -> None:
    render_header()
    _, loaded = render_sidebar()

    tab_health, tab_step5, tab_step7, tab_final, tab_ai = st.tabs(
        [
            "Data Health",
            "Models",
            "Categories",
            "Final",
            "AI Analyst",
        ]
    )

    with tab_health:
        render_data_health(loaded)

    with tab_step5:
        render_step5(loaded)

    with tab_step7:
        render_step7(loaded)

    with tab_final:
        render_final(loaded)

    with tab_ai:
        render_ai_agent(loaded)


if __name__ == "__main__":
    main()
