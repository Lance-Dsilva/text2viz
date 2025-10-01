import streamlit as st
import pandas as pd
import tempfile, os, base64

# ---- LIDA + OpenAI imports ----
LIDA_READY = True
try:
    from lida import Manager
    from llmx import llm
except Exception as e:
    LIDA_READY = False
    INIT_ERROR = e


# ======== LIDA Helpers =========
def run_lida_once(manager: "Manager", csv_path: str, user_goal: str):
    """
    Try generating a visualization once using Matplotlib.
    Returns (raster_image, code_str).
    """
    summary = manager.summarize(csv_path)

    goal = {
        "question": user_goal,
        "visualization": (
            "Generate a single Matplotlib visualization that answers the question. "
            "If necessary, perform data transformations (groupby, filtering, aggregation, resampling weekly/monthly/quarterly)."
        ),
        "rationale": (
            "Explain briefly the reasoning behind the visualization choice and any data transformations performed."
        ),
    }

    charts = manager.visualize(summary=summary, goal=goal, library="matplotlib")
    if not charts:
        return None, ""
    chart = charts[0]
    return getattr(chart, "raster", None), getattr(chart, "code", "")


def run_lida_with_fallbacks(csv_path: str, user_prompt: str) -> tuple:
    """
    Try up to 2 attempts to ensure a visualization is returned.
    """
    manager = Manager(text_gen=llm("openai", model="gpt-4o-mini"))

    # Attempt 1
    raster, code = run_lida_once(manager, csv_path, user_prompt)
    if raster is not None:
        return raster, code, "primary"

    # Attempt 2
    guided_prompt = (
        f"{user_prompt}. Perform any necessary transformations and generate the most meaningful Matplotlib visualization possible."
    )
    raster, code = run_lida_once(manager, csv_path, guided_prompt)
    return raster, code, "fallback" if raster is not None else "none"


def decode_raster(raster):
    """
    Handle raster returned as base64 string or raw bytes.
    """
    if raster is None:
        return None
    if isinstance(raster, bytes):
        return raster
    if isinstance(raster, str):
        try:
            return base64.b64decode(raster)
        except Exception:
            return None
    return None


# ======== Streamlit UI =========

st.set_page_config(page_title="📊 Text → Visualization (LIDA + Matplotlib)", layout="wide")
st.title("📊 Text → Visualization (LIDA + Matplotlib)")

if not LIDA_READY:
    st.error("❌ LIDA is not available. Install with:")
    st.code("pip install lida llmx openai", language="bash")
    st.exception(INIT_ERROR)
    st.stop()

st.write(
    "Upload a dataset, type your question (e.g., `weekly revenue`, `sales by region`, `average profit by month`), "
    "and click **Generate Visualization**. LIDA will use OpenAI's `gpt-4o-mini` to generate Matplotlib code automatically."
)

# Upload section
uploaded = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
sheet = None
if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xls")):
    try:
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select sheet", options=xls.sheet_names, index=0)
        uploaded.seek(0)
    except Exception:
        pass

# Question input
prompt = st.text_input("❓ What visualization would you like to see?", placeholder="e.g., Weekly sales trend by region")

# Enable button only if both inputs exist
can_generate = uploaded is not None and prompt.strip() != ""
gen_clicked = st.button("🚀 Generate Visualization", type="primary", disabled=not can_generate)

if gen_clicked:
    try:
        # Load data
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, sheet_name=sheet if sheet else 0)

        if df.empty:
            st.warning("⚠️ The uploaded file appears empty. Please try a different dataset.")
            st.stop()

        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Save temp CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name

        # Run LIDA
        with st.spinner("🤖 Generating visualization with LIDA + GPT-4o-mini (Matplotlib)..."):
            raster, code, attempt = run_lida_with_fallbacks(csv_path, prompt)

        try:
            os.unlink(csv_path)
        except Exception:
            pass

        # Decode raster if base64
        raster_bytes = decode_raster(raster)

        # Display results
        if raster_bytes is not None:
            st.subheader("📈 Generated Visualization")
            st.image(raster_bytes, caption=f"LIDA Visualization ({attempt} attempt)", use_container_width=True)

            with st.expander("🧠 View Generated Matplotlib Code"):
                st.code(code or "# No code returned", language="python")

            st.caption("✅ Visualization generated using LIDA + OpenAI (`gpt-4o-mini`) with Matplotlib.")
        else:
            st.error("❌ LIDA could not generate a visualization image. Try rephrasing your question or using a different dataset.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
