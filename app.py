import os
import base64
import tempfile

import pandas as pd
import streamlit as st

# =========================
# Page Config & Theme
# =========================
st.set_page_config(
    page_title="DataViz AI",
    page_icon="📊",
    layout="wide",
)

# Global styles (mobile-first)
st.markdown("""
<style>
/* --- Base & layout --- */
:root{
  --brand:#2563eb; /* primary */
  --brand-2:#7c3aed; /* accent */
  --bg:#0b1220;     /* dark blue */
  --card:#0f172a80; /* glass */
  --text:#e5e7eb;   /* light */
  --muted:#94a3b8;  /* muted */
}
html, body, .main { background: radial-gradient(1200px 800px at 10% 0%, #141c2f 0%, #0b1220 45%, #0b1220 100%) !important; }
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1200px; }

/* --- Hero --- */
.hero {
  border-radius: 18px;
  background: linear-gradient(135deg, #1f2937aa, #0f172a66);
  border: 1px solid #1f2a44;
  padding: clamp(16px, 4vw, 28px);
  color: var(--text);
  box-shadow: 0 10px 30px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.03);
  backdrop-filter: blur(10px);
}
.hero h1{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto;
  font-size: clamp(22px, 4.2vw, 36px);
  line-height: 1.15;
  margin: 0 0 .35rem 0;
  color: #f8fafc;
}
.hero .sub{
  color: var(--muted);
  font-size: clamp(13px, 2.6vw, 16px);
  margin: 0;
}

/* --- Cards --- */
.card {
  background: var(--card);
  border: 1px solid #1f2a44;
  border-radius: 16px;
  padding: clamp(12px, 3.6vw, 18px);
  color: var(--text);
  box-shadow: 0 8px 24px rgba(0,0,0,.18);
  backdrop-filter: blur(8px);
}
.card h3{
  margin: 0 0 .75rem 0;
  font-size: clamp(16px, 2.8vw, 18px);
  color: #e2e8f0;
}

/* --- Buttons --- */
.stButton>button {
  width: 100%;
  border: 0;
  border-radius: 10px;
  padding: .75rem 1rem;
  font-weight: 600;
  color: white;
  background: linear-gradient(90deg, var(--brand), var(--brand-2));
  box-shadow: 0 8px 20px rgba(37,99,235,.35);
}
.stButton>button:hover { filter: brightness(1.05); transform: translateY(-1px); }

/* --- Inputs --- */
input[type="text"], .stTextInput>div>div>input {
  border-radius: 10px !important;
  background: #0b1328 !important;
  color: var(--text) !important;
  border: 1px solid #1e293b !important;
}

/* --- Dataframe --- */
[data-testid="stDataFrame"] {
  background: #0b1328 !important;
  border-radius: 10px;
  border: 1px solid #1e293b;
}

/* --- Image --- */
img { border-radius: 12px; border: 1px solid #1f2a44; }

/* --- Badges --- */
.badges {
  display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.5rem
}
.badge {
  background: #0b1328; color:#cbd5e1; border:1px solid #1f2a44; border-radius:999px;
  padding:.25rem .6rem; font-size:.8rem
}

/* --- Footer --- */
.footer {
  color:#94a3b8; font-size:.85rem; margin-top: 1.25rem; text-align:center
}

/* Mobile tweaks */
@media (max-width: 640px){
  .block-container { padding-left: .8rem; padding-right: .8rem; }
}
</style>
""", unsafe_allow_html=True)

# =========================
# Secret handling (Cloud + local)
# =========================
if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =========================
# LIDA + OpenAI imports
# =========================
LIDA_READY = True
try:
    from lida import Manager
    from llmx import llm
except Exception as e:
    LIDA_READY = False
    INIT_ERROR = e

# =========================
# LIDA helpers
# =========================
def run_lida_once(manager: "Manager", csv_path: str, user_goal: str):
    """Single attempt (Matplotlib). Returns (raster, code)."""
    summary = manager.summarize(csv_path)
    goal = {
        "question": user_goal,
        "visualization": (
            "Generate one Matplotlib visualization that answers the question. "
            "If needed, transform data (filter, groupby, aggregation, resample weekly/monthly/quarterly)."
        ),
        "rationale": "Briefly justify the chart and any transformations."
    }
    charts = manager.visualize(summary=summary, goal=goal, library="matplotlib")
    if not charts:
        return None, ""
    c = charts[0]
    return getattr(c, "raster", None), getattr(c, "code", "")

def run_lida(csv_path: str, prompt: str):
    """Two attempts to maximize success."""
    manager = Manager(text_gen=llm("openai", model="gpt-4o-mini"))
    r, code = run_lida_once(manager, csv_path, prompt)
    if r is not None:
        return r, code, "primary"
    guided = f"{prompt}. Perform any required transformations first, then render a single clear Matplotlib chart."
    r, code = run_lida_once(manager, csv_path, guided)
    return r, code, ("fallback" if r is not None else "none")

def decode_raster(raster):
    """Accept bytes or base64 string."""
    if raster is None: return None
    if isinstance(raster, bytes): return raster
    if isinstance(raster, str):
        try: return base64.b64decode(raster)
        except Exception: return None
    return None

# =========================
# Hero / Header
# =========================
st.markdown("""
<div class="hero">
  <h1>DataViz AI</h1>
  <p class="sub">Upload data → Ask in plain English → Get a clean chart. Powered by LIDA + OpenAI (Matplotlib).</p>
  <div class="badges">
    <span class="badge">LIDA</span>
    <span class="badge">OpenAI gpt-4o-mini</span>
    <span class="badge">Matplotlib</span>
    <span class="badge">CSV / Excel</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Guards
# =========================
if not LIDA_READY:
    st.error("LIDA not available. Install:")
    st.code("pip install lida llmx openai", language="bash")
    st.exception(INIT_ERROR)
    st.stop()

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Add your `OPENAI_API_KEY` in Streamlit Cloud → Settings → Secrets.")
    # Do not stop; allow UI exploration

# =========================
# Input Row (compact, minimal text)
# =========================
left, right = st.columns([1,1], vertical_alignment="bottom")

with left:
    up_card = st.container()
    with up_card:
        st.markdown('<div class="card"><h3>📤 Upload</h3>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV / Excel", type=["csv","xlsx","xls"], label_visibility="collapsed")
        sheet = None
        if uploaded is not None and uploaded.name.lower().endswith((".xlsx",".xls")):
            try:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Sheet", options=xls.sheet_names, index=0, label_visibility="collapsed")
                uploaded.seek(0)
            except Exception:
                pass
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    q_card = st.container()
    with q_card:
        st.markdown('<div class="card"><h3>❓ Question</h3>', unsafe_allow_html=True)
        prompt = st.text_input(
            "Ask",
            placeholder="e.g., Weekly revenue trend by region",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

# CTA
can_generate = uploaded is not None and prompt.strip() != ""
cta_col = st.container()
with cta_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    gen_clicked = st.button("🚀 Generate Visualization", disabled=not can_generate)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Result
# =========================
if gen_clicked:
    try:
        # Load df
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, sheet_name=sheet if sheet else 0)

        if df.empty:
            st.error("File is empty. Try a different dataset.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Temp CSV for LIDA
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                csv_path = tmp.name

            with st.spinner("Generating chart…"):
                raster, code, attempt = run_lida(csv_path, prompt)

            # Cleanup
            try: os.unlink(csv_path)
            except Exception: pass

            raster_bytes = decode_raster(raster)

            st.markdown("### 📈 Visualization")
            if raster_bytes is not None:
                st.image(raster_bytes, caption=f"LIDA ({attempt})", use_container_width=True)
                with st.expander("View generated Matplotlib code"):
                    st.code(code or "# No code returned", language="python")
                st.success("Done.")
            else:
                st.error("Couldn’t generate an image. Try a clearer question or a dataset with numeric columns.")

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# Footer (minimal)
# =========================
st.markdown('<div class="footer">© DataViz AI — built with LIDA + OpenAI • Optimized for mobile</div>', unsafe_allow_html=True)
