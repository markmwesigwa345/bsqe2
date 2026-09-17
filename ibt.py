"""
ibt.py — BSQE2 AI Study Assistant
─────────────────────────────────────────────────────────────────────────────
Multi-subject RAG chatbot for Bachelor of Science in Quantitative Economics (BSQE2) students.

Architecture:
  - FAISS vector stores (FAISS native format) loaded per-subject at runtime
  - HuggingFace all-MiniLM-L6-v2 for dense retrieval
  - Gemini 2.5 Flash as the generative LLM
  - LangChain LCEL pipeline (Runnable-based, no legacy chains)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import re
import uuid
import time
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv


# MUST be the very first Streamlit call
st.set_page_config(
    page_title="BSQE2 AI Study Assistant 🎓",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Imports (with clear error reporting) ──────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.runnables import RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.chains import create_history_aware_retriever
except ImportError as e:
    st.error(f"❌ Missing dependency: **{e}**")
    st.info(
        "Run `pip install -r requirements.txt` to install all required packages, "
        "then restart the app."
    )
    st.stop()

try:
    from langchain_groq import ChatGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ── Subject Registry ──────────────────────────────────────────────────────────
from subjects_config import SUBJECTS

# ── Paths & Session Persistence Storage ───────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(SCRIPT_DIR, ".chat_sessions")

def _ensure_sessions_dir():
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
    except Exception:
        pass


def load_session_history(session_id: str) -> dict:
    """Load chat history dictionary across all subjects for a given session ID (fail-safe)."""
    _ensure_sessions_dir()
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def save_session_history(session_id: str, history_dict: dict):
    """Persist chat history dictionary across all subjects to disk (fail-safe)."""
    _ensure_sessions_dir()
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ── Environment / API Key ─────────────────────────────────────────────────────
# Priority: Streamlit Cloud secrets → .env file → environment variable
load_dotenv()
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = str(st.secrets["GOOGLE_API_KEY"]).strip().strip('"').strip("'")
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"]).strip().strip('"').strip("'")
except Exception:
    pass  # No secrets.toml found; fall back to .env

if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY").strip().strip('"').strip("'")
else:
    st.error("⚠️ **GOOGLE_API_KEY not found!**")
    st.info(
        "**Local:** Add `GOOGLE_API_KEY=your-key` to `.env` or "
        "`.streamlit/secrets.toml`.\n\n"
        "**Streamlit Cloud:** Go to App → Settings → Secrets and add your key."
    )
    st.stop()

# ── Theme Definition (single source of truth) ─────────────────────────────────
# Every color used anywhere in the app is pulled from this dict, keyed by mode.
# This replaces the old scattered variables + the non-functional st._config calls
# (st._config.set_option is a private/internal API that does not reliably
# re-theme an already-running Streamlit session, so it has been removed).
THEME = {
    "dark": {
        "bg":           "#0E1117",
        "sidebar_bg":   "#161B22",
        "text":         "#FAFAFA",
        "text_muted":   "#9CA3AF",
        "card":         "#1E232A",
        "input_bg":     "#262730",
        "border":       "#30363D",
        "accent":       "#2563EB",
        "accent_text":  "#FFFFFF",
    },
    "light": {
        "bg":           "#FFFFFF",
        "sidebar_bg":   "#F8FAFC",
        "text":         "#0F172A",
        "text_muted":   "#64748B",
        "card":         "#F1F5F9",
        "input_bg":     "#FFFFFF",
        "border":       "#CBD5E1",
        "accent":       "#2563EB",
        "accent_text":  "#FFFFFF",
    },
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("BSQE2 AI Assistant 📊")

    is_dark = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode,
        key="dark_mode_toggle",
    )
    st.session_state.dark_mode = is_dark
    st.divider()

    st.markdown("### 📚 Choose a Course")
    subject_names = list(SUBJECTS.keys())
    selected_subject_name = st.radio(
        "Select a course unit to study:",
        subject_names,
        index=0,
    )

    selected_cfg = SUBJECTS[selected_subject_name]

    st.divider()
    st.markdown(f"**About {selected_subject_name}:**")
    st.caption(selected_cfg["description"])
    st.divider()
    if st.button("🗑️ Clear History for this Course", use_container_width=True):
        if "subject_messages" in st.session_state and selected_subject_name in st.session_state.subject_messages:
            st.session_state.subject_messages[selected_subject_name] = []
            if "session_id" in st.session_state:
                save_session_history(st.session_state.session_id, st.session_state.subject_messages)
            st.rerun()
    if st.button("⚠️ Clear ALL Course Histories", use_container_width=True):
        st.session_state.subject_messages = {}
        if "session_id" in st.session_state:
            save_session_history(st.session_state.session_id, {})
        st.rerun()
    st.divider()

    # ── Test Me Mode toggle ───────────────────────────────────────────────────
    prev_test_me = st.session_state.get("test_me_mode", False)
    test_me_on = st.toggle("🧠 Test Me Mode", value=prev_test_me, key="test_me_toggle")
    if test_me_on and st.session_state.get("quiz_mode"):
        st.session_state.quiz_mode = False
    st.session_state.test_me_mode = test_me_on

    if test_me_on:
        score = st.session_state.get("test_me_score", {"correct": 0, "total": 0, "sum": 0})
        if score["total"] > 0:
            avg = score["sum"] // score["total"]
            st.caption(f"🧠 {score['total']} question(s) — Avg: {avg}/100")
        else:
            st.caption("🧠 No questions answered yet")

    if prev_test_me and not test_me_on:
        score = st.session_state.get("test_me_score", {"correct": 0, "total": 0, "sum": 0})
        if score["total"] > 0:
            avg = score["sum"] // score["total"]
            st.success(f"**Test Me Session Complete!**\n\nQuestions: {score['total']} | Average Score: {avg}/100")
        st.session_state.test_me_score = {"correct": 0, "total": 0, "sum": 0}
        st.session_state.pop("test_me_question", None)
        st.session_state.pop("test_me_context", None)

    st.divider()

    # ── Quiz Mode toggle ──────────────────────────────────────────────────────
    prev_quiz = st.session_state.get("quiz_mode", False)
    quiz_on = st.toggle("📝 Quiz Mode (MCQ)", value=prev_quiz, key="quiz_mode_toggle")
    if quiz_on and st.session_state.get("test_me_mode"):
        st.session_state.test_me_mode = False
    st.session_state.quiz_mode = quiz_on

    if quiz_on:
        qs = st.session_state.get("mcq_score", {"total": 0, "sum": 0})
        if qs["total"] > 0:
            avg = qs["sum"] // qs["total"]
            st.caption(f"📝 {qs['total']} question(s) — Avg: {avg}/100")
        else:
            st.caption("📝 No questions answered yet")

    if prev_quiz and not quiz_on:
        qs = st.session_state.get("mcq_score", {"total": 0, "sum": 0})
        if qs["total"] > 0:
            avg = qs["sum"] // qs["total"]
            st.success(f"**Quiz Session Complete!**\n\nQuestions: {qs['total']} | Average Score: {avg}/100")
        st.session_state.mcq_score = {"total": 0, "sum": 0}
        st.session_state.pop("mcq_question", None)

    st.divider()
    st.write("Made by Mwesigwa Mark")

# Resolve active theme dict for this run
t = THEME["dark" if is_dark else "light"]

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    /* CSS Custom Properties for theme variables */
    :root {{
        --bg-main: {t["bg"]};
        --sidebar-bg: {t["sidebar_bg"]};
        --text-main: {t["text"]};
        --text-muted: {t["text_muted"]};
        --card-bg: {t["card"]};
        --input-bg: {t["input_bg"]};
        --border-color: {t["border"]};
        --accent-color: {t["accent"]};
        --accent-text: {t["accent_text"]};
    }}

    /* Base app container and main page pane */
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stMain"], 
    [data-testid="stMainBlockContainer"], 
    [data-testid="stAppViewBlockContainer"] {{
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }}

    /* Sidebar container */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        color: var(--text-main) !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-main);
    }}

    /* Header toolbar (Top Right buttons / icons) */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    header[data-testid="stHeader"] *,
    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] svg path {{
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
    }}

    /* Top-right menu popover / dropdowns */
    [data-testid="stPopoverBody"], div[role="listbox"], div[role="menu"] {{
        background-color: var(--card-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
    }}

    /* Radio buttons & Toggles accent overrides */
    [data-testid="stRadio"] label p,
    [data-testid="stCheckbox"] label p {{
        color: var(--text-main) !important;
    }}
    /* Radio selection circle / dot / border styling */
    [data-testid="stRadio"] *, 
    [data-baseweb="radio"] *, 
    [data-baseweb="checkbox"] * {{
        accent-color: var(--accent-color) !important;
    }}
    [aria-checked="true"] > div,
    [data-baseweb="radio"] [aria-checked="true"] div,
    [data-testid="stRadio"] [aria-checked="true"] div {{
        background-color: var(--accent-color) !important;
        border-color: var(--accent-color) !important;
    }}
    [data-testid="stRadio"] svg,
    [data-baseweb="radio"] svg {{
        fill: var(--accent-color) !important;
        color: var(--accent-color) !important;
    }}
    /* Widget labels & Captions */
    [data-testid="stWidgetLabel"] p, label p {{
        color: var(--text-main) !important;
        font-weight: 500;
    }}
    [data-testid="stCaptionContainer"], .stCaption {{
        color: var(--text-muted) !important;
    }}

    /* Radio selection dot & border styling */
    [data-testid="stRadio"] div[role="radiogroup"] [aria-checked="true"] svg {{
        fill: var(--accent-color) !important;
    }}
    [data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] div:first-child {{
        border-color: var(--accent-color) !important;
    }}
    [data-baseweb="radio"] [aria-checked="true"] > div {{
        background-color: var(--accent-color) !important;
        border-color: var(--accent-color) !important;
    }}
    [data-baseweb="checkbox"] [aria-checked="true"] > div {{
        background-color: var(--accent-color) !important;
        border-color: var(--accent-color) !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] > div {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stAlert"] p, [data-testid="stAlert"] span {{
        color: var(--text-main) !important;
    }}
    [data-testid="stAlert"] svg {{
        fill: var(--accent-color) !important;
    }}

    /* Hide native yellow spinner — make SVG stroke transparent, hide label */
    [data-testid="stSpinner"] svg circle {{
        stroke: transparent !important;
    }}
    [data-testid="stSpinner"] > div > p {{
        display: none !important;
    }}

    /* Chat message cards */
    [data-testid="stChatMessage"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }}
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] h5,
    [data-testid="stChatMessage"] h6,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {{
        color: var(--text-main) !important;
    }}
    [data-testid="stChatMessage"] code {{
        background-color: var(--input-bg) !important;
        color: var(--text-main) !important;
    }}

    /* Vibrant BSQE2 avatar badge — replaces Streamlit's default icon
       in place. font-size:0 blanks out ANY native content (emoji,
       text, image) so nothing leaks through behind our own label. */
    [data-testid="stChatMessageAvatarAssistant"] {{
        background: linear-gradient(135deg, {t["accent"]} 0%, #1e40af 100%) !important;
        border: none !important;
        border-radius: 50% !important;
        box-shadow:
            0 2px 10px rgba(37, 99, 235, 0.45),
            0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        font-size: 0 !important;
        color: transparent !important;
    }}
    [data-testid="stChatMessageAvatarAssistant"] * {{
        display: none !important;
    }}
    [data-testid="stChatMessageAvatarAssistant"]::after {{
        content: "BSQE2";
        display: block;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1;
        letter-spacing: 0.3px;
        white-space: nowrap;
        animation: bsqe2-badge-glow 2.4s ease-in-out infinite;
    }}

    /* Safety net: hide any second avatar variant Streamlit renders in the same row */
    [data-testid="stChatMessageAvatarAssistant"] ~ [data-testid="stChatMessageAvatarAssistant"] {{
        display: none !important;
    }}

    @keyframes bsqe2-badge-glow {{
        0%, 100%  {{ text-shadow: 0 0 3px rgba(255,255,255,0.25); opacity: 0.92; }}
        50%       {{ text-shadow: 0 0 8px rgba(255,255,255,0.9);  opacity: 1;    }}
    }}
    @keyframes bsqe2-badge-pulse-active {{
        0%, 100% {{ transform: scale(1);    box-shadow: 0 2px 10px rgba(37,99,235,0.45), 0 0 0 3px rgba(37,99,235,0.12); }}
        50%      {{ transform: scale(1.12); box-shadow: 0 2px 16px rgba(37,99,235,0.7), 0 0 0 6px rgba(37,99,235,0.18); }}
    }}
    .stChatMessage:has([data-testid="stSpinner"]) [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarAssistant"]:has(+ * [data-testid="stSpinner"]) {{
        animation: bsqe2-badge-pulse-active 1s ease-in-out infinite !important;
    }}
    @keyframes bsqe2-badge-pulse-active {{
        0%, 100% {{ transform: scale(1);    box-shadow: 0 2px 10px rgba(37,99,235,0.45), 0 0 0 3px rgba(37,99,235,0.12); }}
        50%      {{ transform: scale(1.12); box-shadow: 0 2px 16px rgba(37,99,235,0.7), 0 0 0 6px rgba(37,99,235,0.18); }}
    }}
    .stChatMessage:has([data-testid="stSpinner"]) [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarAssistant"]:has(+ * [data-testid="stSpinner"]) {{
        animation: bsqe2-badge-pulse-active 1s ease-in-out infinite !important;
    }}

    /* Markdown Tables (Fix for dark text/low contrast on tables inside chat) */
    [data-testid="stChatMessage"] table,
    .stMarkdown table {{
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 12px 0 !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] td,
    .stMarkdown th,
    .stMarkdown td {{
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        padding: 8px 12px !important;
        text-align: left !important;
        vertical-align: top !important;
    }}
    [data-testid="stChatMessage"] th,
    .stMarkdown th {{
        background-color: var(--input-bg) !important;
        font-weight: 600 !important;
    }}
    [data-testid="stChatMessage"] tr:nth-child(even),
    .stMarkdown tr:nth-child(even) {{
        background-color: rgba(128, 128, 128, 0.08) !important;
    }}

    /* Bottom container (Chat Input area & sticky bar) */
    [data-testid="stBottom"], 
    [data-testid="stBottomBlockContainer"],
    .stApp > footer {{
        background-color: var(--bg-main) !important;
        border-top: 1px solid var(--border-color) !important;
    }}
    
    /* Strip backgrounds from all intermediate wrapper/spacer elements so they are transparent */
    [data-testid="stBottom"] *,
    [data-testid="stBottomBlockContainer"] * {{
        background-color: transparent !important;
    }}

    /* Target Chat Input Box */
    [data-testid="stBottom"] [data-testid="stChatInput"] {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: var(--text-main) !important;
        caret-color: var(--text-main) !important;
        -webkit-text-fill-color: var(--text-main) !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
        opacity: 1 !important;
    }}

    /* Chat submit button: scoped accent style */
    [data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] {{
        background-color: var(--accent-color) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stBottom"] [data-testid="stChatInputSubmitButton"] svg {{
        fill: var(--accent-text) !important;
    }}

    /* Buttons */
    [data-testid="stBaseButton-secondary"] {{
        background-color: var(--input-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stBaseButton-secondary"]:hover {{
        border-color: var(--accent-color) !important;
        color: var(--accent-color) !important;
    }}
    [data-testid="stBaseButton-primary"] {{
        background-color: {t["accent"]} !important;
        color: {t["accent_text"]} !important;
        border: 1px solid {t["accent"]} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stBaseButton-primary"] p {{
        color: {t["accent_text"]} !important;
    }}

    /* Inline code / markdown code blocks outside chat */
    .stMarkdown code {{
        background-color: {t["input_bg"]} !important;
        color: {t["text"]} !important;
    }}

    /* Subject badge: its own explicit color pair */
    .subject-badge {{
        display: inline-block;
        background: {t["accent"]};
        color: {t["accent_text"]} !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
        margin-bottom: 12px;
    }}

    hr {{
        border-color: {t["border"]} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session & Storage Persistence (Survives Refresh & Subject Switching) ──────
if "session_id" not in st.query_params:
    session_id = str(uuid.uuid4())[:8]
    st.query_params["session_id"] = session_id
else:
    session_id = st.query_params["session_id"]

st.session_state.session_id = session_id

if "subject_messages" not in st.session_state:
    st.session_state.subject_messages = load_session_history(session_id)

if selected_subject_name not in st.session_state.subject_messages:
    st.session_state.subject_messages[selected_subject_name] = []

current_messages = st.session_state.subject_messages[selected_subject_name]

# ── Intent Classifier (Bypass FAISS Retrieval for Casual Queries) ─────────────

def is_conversational_query(text: str) -> bool:
    """
    Returns True if the text is a greeting, farewell, or casual pleasantry
    that does NOT require document retrieval from FAISS.

    Uses re.fullmatch so patterns must cover the ENTIRE input — prevents
    short academic queries like "Hi explain MRS" from being misclassified.
    """
    cleaned = text.strip().lower()

    # Never treat summarization requests as casual greetings
    if any(kw in cleaned for kw in ["summarize", "summary", "brief", "recap", "short version", "bullet points"]):
        return False

    # ── Exact-match casual phrases ────────────────────────────────────────────
    casual_phrases = {
        # Greetings
        "hi", "hi there", "hello", "hello there", "hey", "hey there", "hey!",
        "howdy", "hiya", "what's up", "whats up", "sup", "yo",
        "good morning", "good afternoon", "good evening", "good day", "good night",
        "greetings", "salutations",

        # Appreciation / politeness
        "thanks", "thank you", "thank you so much", "thanks a lot", "thanks a bunch",
        "cheers", "much appreciated", "appreciate it", "ty", "thx",

        # Farewells
        "bye", "goodbye", "see you", "see ya", "later", "take care",
        "have a good day", "have a great day", "ttyl", "talk later",

        # Identity / meta questions
        "who are you", "who created you", "who made you", "who built you",
        "who built this", "who built this app", "who made this app",
        "what are you", "what can you do", "what do you do",
        "are you an ai", "are you a bot", "are you human",
        "tell me about yourself",

        # Affirmations / small talk
        "ok", "okay", "ok thanks", "okay thanks", "got it", "i see",
        "cool", "nice", "awesome", "great", "perfect", "sounds good",
        "sure", "alright", "no problem", "no worries",
    }
    if cleaned in casual_phrases:
        return True

    # ── Pattern matching (fullmatch — must cover the ENTIRE input) ────────────
    # Prevents "Hi explain MRS" from being misclassified as a greeting
    greeting_patterns = [
        # Simple greetings ± punctuation/whitespace
        r"(hi|hello|hey|howdy|hiya|yo|sup|greetings|salutations)[\!\?\.,\s]*",
        # Time-of-day greetings
        r"good\s(morning|afternoon|evening|day|night)[\!\?\.,\s]*",
        # Farewells
        r"(bye|goodbye|see\s(you|ya)|later|take\s?care|ttyl)[\!\?\.,\s]*",
        # Thanks
        r"(thank\s?you|thanks|cheers|ty|thx)[\!\?\.,\s]*",
        # Identity / meta
        r"(who\s(are|created|made|built)\s(you|this(\s?app)?))[\?\.,\s]*",
        r"(what\s(are|can)\syou\s?(do)?)[\?\.,\s]*",
        r"(are\syou\s(an?\s)?(ai|bot|human))[\?\.,\s]*",
        # Short affirmations
        r"(ok|okay|got\sit|i\ssee|cool|nice|awesome|great|perfect|alright|sure)[\!\?\.,\s]*",
    ]
    for pattern in greeting_patterns:
        if re.fullmatch(pattern, cleaned):
            # If more than 3 words total, it's likely a real question — send to retrieval
            if len(cleaned.split()) > 3:
                return False
            return True

    return False


def is_summarization_request(text: str) -> bool:
    """Detects if the user prompt is asking to summarize a previous answer or topic."""
    cleaned = text.strip().lower()
    keywords = [
        "summarize", "summary", "briefly explain", "in short",
        "bullet points", "give me a summary", "key takeaways", "recap",
        "shorten this", "summarise"
    ]
    return any(kw in cleaned for kw in keywords)


# ── Cached Resource Loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model (cached globally)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatGoogleGenerativeAI:
    """Load and cache the primary Gemini LLM."""
    return ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=1,
        timeout=20,
    )


@st.cache_resource(show_spinner=False)
def load_llm_gemini_fallback() -> ChatGoogleGenerativeAI:
    """Gemini 3.5 Flash Lite — lighter Gemini model for MCQ/flashcard generation."""
    return ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=1,
        timeout=20,
    )


@st.cache_resource(show_spinner=False)
def load_groq_llm_primary():
    """Groq compound-mini — primary Groq model for grading (fast, within token limits)."""
    if not _GROQ_AVAILABLE:
        return None
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        return None
    try:
        return ChatGroq(model="groq/compound-mini", temperature=0.1, groq_api_key=key, max_retries=1)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_groq_llm_secondary():
    """Groq qwen3.8-27b — secondary Groq model for question generation."""
    if not _GROQ_AVAILABLE:
        return None
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        return None
    try:
        return ChatGroq(model="qwen/qwen3.8-27b", temperature=0.1, groq_api_key=key, max_retries=1)
    except Exception:
        return None


def get_available_llm():
    """Return ordered list of LLMs to try: Gemini primary → Gemini fallback → Groq x2."""
    candidates = [
        load_llm(),
        load_llm_gemini_fallback(),
        load_groq_llm_primary(),
        load_groq_llm_secondary(),
    ]
    return [m for m in candidates if m is not None]


@st.cache_resource(show_spinner=False)
def load_vector_store(faiss_dir: str, subject_name: str):
    """
    Load a FAISS vector store from a native FAISS directory.

    The directory must have been created by `FAISS.save_local()` in the
    Colab notebook (contains index.faiss + index.pkl).

    Returns:
        (FAISS, None)       on success
        (None, error_str)   on failure
    """
    abs_path = (
        faiss_dir if os.path.isabs(faiss_dir)
        else os.path.join(SCRIPT_DIR, os.path.normpath(faiss_dir))
    )

    if not os.path.isdir(abs_path):
        available = [
            d for d in os.listdir(SCRIPT_DIR)
            if os.path.isdir(os.path.join(SCRIPT_DIR, d)) and d.endswith("_faiss")
        ]
        return None, (
            f"❌ Vector store folder **`{faiss_dir}`** not found for **{subject_name}**.\n\n"
            f"📂 Available FAISS folders: `{available or 'none yet'}`\n\n"
            "💡 Run the Colab notebook for this subject to generate the index folder, "
            "then place it in the project directory."
        )

    try:
        embeddings = load_embeddings()
        vector_store = FAISS.load_local(
            folder_path=abs_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,  # Required by LangChain ≥0.3
        )
        return vector_store, None
    except Exception as exc:
        return None, (
            f"❌ Failed to load vector store for **{subject_name}**: `{exc}`\n\n"
            "💡 Re-generate the FAISS folder using the updated `colab.ipynb` notebook "
            "and replace the existing folder."
        )


@st.cache_resource(show_spinner=False)
def setup_qa_components(_vector_store, subject_prompt: str) -> dict:
    """Build and cache the QA retriever, prompt, and LLM components for the given vector store.

    Reformulation (query rewriting) uses compound-mini — fast, lightweight, no deep reasoning needed.
    Answer generation uses gemini-2.0-flash — full academic depth reserved for the actual answer.
    """
    answer_llm = load_llm()  # gemini-2.0-flash — deep academic answers
    reformat_llm = load_groq_llm_secondary() or load_llm()  # compound-mini — fast query rewriting

    retriever = _vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10},
    )

    reformulation_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", (
            "Given the conversation above, rephrase my latest message into a single, "
            "complete, self-contained question that captures exactly what I am asking. "
            "Do not answer it — only rewrite it."
        )),
    ])

    history_aware_retriever = create_history_aware_retriever(
        reformat_llm, retriever, reformulation_prompt
    )

    answer_prompt = ChatPromptTemplate.from_template(subject_prompt)

    return {
        "retriever": history_aware_retriever,
        "prompt": answer_prompt,
        "llm": answer_llm,
    }


def extract_citation_metadata(docs: list) -> list:
    """Format metadata from retrieved Document objects into citation data for UI rendering."""
    citations = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        raw_src = meta.get("source", "Course Materials")
        src_name = os.path.basename(str(raw_src))
        page_num = meta.get("page") if meta.get("page") is not None else meta.get("page_number")
        
        if isinstance(page_num, int):
            page_str = f" (Page {page_num + 1})"
        elif page_num:
            page_str = f" (Page {page_num})"
        else:
            page_str = ""
            
        snippet = doc.page_content[:280].strip() + ("…" if len(doc.page_content) > 280 else "")
        citations.append({
            "source": src_name,
            "page": page_str,
            "snippet": snippet,
        })
    return citations


def build_chat_history(messages: list) -> list:
    """Convert st.session_state.messages into LangChain HumanMessage/AIMessage
    objects so the history-aware retriever can read conversation context."""
    history = []
    for msg in messages[-6:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history



def show_thinking_indicator(placeholder):
    placeholder.markdown(
        """
        <span style="color:#9CA3AF;font-size:13px;font-style:italic;">
            thinking…
        </span>
        """,
        unsafe_allow_html=True,
    )


_CONTEXT_REF_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her"
    r"|the above|the previous|you mentioned|you said|elaborate|expand|more detail"
    r"|explain more|tell me more|what about|how about|why is that|what does that mean)"
    r"\b",
    re.IGNORECASE,
)

# ── Query Cleaner ────────────────────────────────────────────────────────────
_FILLER_RE = re.compile(
    r"\b(please|can you|could you|i want to|i need to|i would like to"
    r"|tell me|explain to me|help me|kindly|just|basically|actually"
    r"|i was wondering|would you mind|do you know|can you please)"
    r"\b",
    re.IGNORECASE,
)

_ABBREV_MAP = {
    r"\bMRS\b": "Marginal Rate of Substitution",
    r"\bMRT\b": "Marginal Rate of Transformation",
    r"\bMPC\b": "Marginal Propensity to Consume",
    r"\bMPS\b": "Marginal Propensity to Save",
    r"\bGDP\b": "Gross Domestic Product",
    r"\bGNP\b": "Gross National Product",
    r"\bCPI\b": "Consumer Price Index",
    r"\bPPC\b": "Production Possibility Curve",
    r"\bIS\b":  "Investment Savings",
    r"\bLM\b":  "Liquidity Money",
    r"\bOLS\b": "Ordinary Least Squares",
    r"\bIV\b":  "Instrumental Variable",
    r"\bVAR\b": "Vector Autoregression",
    r"\bNPV\b": "Net Present Value",
    r"\bIRR\b": "Internal Rate of Return",
    r"\bROI\b": "Return on Investment",
    r"\bEBIT\b": "Earnings Before Interest and Tax",
    r"\bSWOT\b": "Strengths Weaknesses Opportunities Threats",
}

def clean_query(text: str) -> str:
    """Strip filler phrases and expand subject-specific abbreviations before FAISS retrieval."""
    cleaned = _FILLER_RE.sub("", text).strip()
    for pattern, expansion in _ABBREV_MAP.items():
        cleaned = re.sub(pattern, expansion, cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned or text


def _parse_score_json(raw: str) -> dict:
    """Robustly extract {score, feedback} JSON from Gemini output.
    Handles prose wrapping, code fences, and single-quoted JSON."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw.strip())
    # Try direct parse first
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Extract first {...} block from prose
    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # Last resort: pull score and feedback with regex
    score_m = re.search(r'"?score"?\s*:\s*(\d+)', raw)
    feedback_m = re.search(r'"?feedback"?\s*:\s*"([^"]+)"', raw)
    if score_m:
        return {
            "score": int(score_m.group(1)),
            "feedback": feedback_m.group(1) if feedback_m else ""
        }
    raise ValueError(f"Could not parse score JSON from: {raw[:200]}")


def generate_mcq(vector_store, llm) -> dict:
    """Pull a random chunk from FAISS and ask LLM to produce one MCQ.
    Returns {question, options:{A,B,C,D}, answer, explanation} or raises on failure."""
    import random
    seed_terms = ["definition", "concept", "theory", "formula", "method", "model", "principle"]
    docs = vector_store.similarity_search(random.choice(seed_terms), k=5)
    if not docs:
        return {}
    chunk = random.choice(docs).page_content[:600]
    prompt = (
        "From the course content below, create ONE multiple-choice question with exactly 4 options (A, B, C, D). "
        "One option must be correct. The other three should be plausible but wrong. "
        "Respond ONLY with valid JSON — no extra text:\n"
        '{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
        '"answer": "<letter>", "explanation": "<one sentence>"}\n\nContent:\n' + chunk
    )
    raw = (llm | StrOutputParser()).invoke(prompt)
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw.strip())
    data = json.loads(raw)
    if all(k in data for k in ("question", "options", "answer", "explanation")):
        return data
    return {}


def score_badge(score: int) -> str:
    """Return a colour-coded score badge string."""
    if score >= 75:
        return f"🟢 {score}/100 — Strong answer!"
    elif score >= 50:
        return f"🟡 {score}/100 — Partially correct."
    else:
        return f"🔴 {score}/100 — Needs review."


def generate_flashcards(docs: list, llm) -> list:
    """Ask LLM to produce 3-5 Q&A flashcard pairs from retrieved chunks.
    Returns list of {question, answer} dicts."""
    context = "\n\n".join(doc.page_content[:400] for doc in docs[:4])
    prompt = (
        "You are a study assistant. From the content below, create exactly 4 flashcard pairs."
        " Respond ONLY with valid JSON — a list of objects with keys 'question' and 'answer'."
        " Each answer should be 1-3 sentences. No extra text outside the JSON.\n\nContent:\n" + context
    )
    try:
        raw = (llm | StrOutputParser()).invoke(prompt)
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"```$", "", raw.strip())
        cards = json.loads(raw)
        return [c for c in cards if "question" in c and "answer" in c][:5]
    except Exception:
        return []


def generate_followups(docs: list) -> list:
    """Extract up to 3 follow-up question suggestions from retrieved chunk content.
    Uses capitalized noun phrases (2-4 words) as key terms — no LLM call."""
    text = " ".join(doc.page_content for doc in docs[:3])
    # Match capitalized multi-word phrases (likely concepts/terms)
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text)
    # Deduplicate while preserving order
    seen, unique = set(), []
    for c in candidates:
        if c.lower() not in seen and len(c) > 6:
            seen.add(c.lower())
            unique.append(c)
    # Build question strings from top 3 unique terms
    questions = []
    templates = ["What is {}?", "How does {} work?", "Explain {} in detail."]
    for i, term in enumerate(unique[:3]):
        questions.append(templates[i % len(templates)].format(term))
    return questions


def needs_reformulation(text: str, has_history: bool) -> bool:
    """Return True only when the query references prior context and a reformulation
    LLM call is actually needed. Avoids the extra Gemini round-trip for fresh questions."""
    if not has_history:
        return False
    return bool(_CONTEXT_REF_RE.search(text))


def stream_with_retry(chain, prompt_input, placeholder, max_retries: int = 2, base_delay: int = 1):
    """Stream into a Streamlit placeholder with throttled DOM updates (every 20 tokens).
    On 429/quota errors, swaps to the next model in the pool instead of retrying the same one.
    On transient 503/overload errors, retries with exponential backoff.
    Returns the full response string."""
    # Extract the parser from the chain if it's a pipeline (llm | parser)
    # so we can swap just the LLM part when falling back
    last_exc = None

    # Determine if this chain is a simple llm|parser pipeline we can swap
    # For non-swappable chains (e.g. history-aware retriever), just retry normally
    llm_pool = get_available_llm()
    parser = StrOutputParser()

    # Try each model in the pool for quota errors
    for model_idx, llm in enumerate(llm_pool):
        current_chain = llm | parser if not hasattr(chain, 'steps') else chain
        # If caller passed a pre-built non-swappable chain, only use it on first model
        if model_idx > 0 and hasattr(chain, 'steps'):
            break


        for attempt in range(max_retries):
            try:
                full = ""
                buf = 0
                for chunk in current_chain.stream(prompt_input):
                    full += chunk
                    buf += 1
                    if buf == 1 or buf % 20 == 0:
                        placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                return full
            except Exception as e:
                msg = str(e).lower()
                last_exc = e
                if any(code in msg for code in ["429", "resource_exhausted", "quota", "rate_limit", "rate limit"]):
                    # Quota hit — break inner loop, try next model
                    break
                elif any(code in msg for code in ["503", "unavailable", "overloaded", "deadline"]):
                    if attempt < max_retries - 1:
                        wait = base_delay * (2 ** attempt)
                        time.sleep(wait)
                        continue
                    break
                else:
                    raise
    raise last_exc

# ── Main App ──────────────────────────────────────────────────────────────────

st.title("BSQE2 AI")
st.markdown(
    f"<div class='subject-badge'>{selected_cfg['icon']} {selected_subject_name}</div>",
    unsafe_allow_html=True,
)

# Load resources for the currently selected subject
with st.spinner(f"Loading resources for **{selected_subject_name}**…"):
    try:
        vector_store, load_error = load_vector_store(
            selected_cfg["faiss_dir"], selected_subject_name
        )
    except Exception as exc:
        load_error = f"❌ Unexpected error: {exc}"
        vector_store = None

if load_error:
    st.error(load_error)
    st.warning(
        f"Please ensure the folder **`{selected_cfg['faiss_dir']}`** "
        "is present in the project directory."
    )
    st.stop()

try:
    qa_setup = setup_qa_components(vector_store, selected_cfg["prompt"])
    if not isinstance(qa_setup, dict):
        st.cache_resource.clear()
        st.rerun()
except Exception as exc:
    st.error(f"❌ Failed to set up the AI chain: {exc}")
    st.stop()

# Welcome message (shown only when the chat is empty)
if not current_messages:
    st.success(
        f"Hi there 📊 I am **BSQE AI**, your study assistant for Bachelor of Science in Quantitative Economics. "
        f"Ready to assist with **{selected_subject_name}**. Ask any question below!"
    )

# ── Chat History ──────────────────────────────────────────────────────────────
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Source Citations & Passages"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**[{idx}] {src['source']}{src['page']}**")
                    st.caption(f"\"{src['snippet']}\"")

# ── Chat Input & Response ─────────────────────────────────────────────────────
MAX_INPUT_CHARS = 1500

if user_prompt := st.chat_input(f"Ask a question about {selected_subject_name}…"):
    if len(user_prompt) > MAX_INPUT_CHARS:
        st.warning(
            f"⚠️ **Question too long** ({len(user_prompt):,} characters). "
            f"Please limit your question to under {MAX_INPUT_CHARS:,} characters for best results."
        )
        st.stop()

    current_messages.append({"role": "user", "content": user_prompt})
    save_session_history(session_id, st.session_state.subject_messages)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        current_sources = []
        show_thinking_indicator(message_placeholder)

        try:
            # PATH A: Casual Greetings / Pleasantries — compound-mini (fast, lightweight)
            if is_conversational_query(user_prompt):
                llm = load_groq_llm_secondary() or load_llm()
                greeting_prompt = (
                    f"You are BSQE2 AI, an elite study assistant for Bachelor of Science in Quantitative Economics students "
                    f"currently helping with {selected_subject_name}.\n\n"
                    f"Respond warmly, naturally, and concisely to this user greeting: '{user_prompt}'."
                )
                full_response = stream_with_retry(llm | StrOutputParser(), greeting_prompt, message_placeholder)

            # PATH B: Explicit User Summarization Request — gemini-2.0-flash (structured, concise)
            elif is_summarization_request(user_prompt):
                llm = load_llm()
                last_assistant_msg = None
                for msg in reversed(current_messages[:-1]):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg["content"]
                        break

                if last_assistant_msg:
                    summary_prompt = (
                        f"You are BSQE2 AI. Summarize the following answer clearly into concise bullet points, "
                        f"highlighting key definitions and main takeaways:\n\n{last_assistant_msg}"
                    )
                    full_response = stream_with_retry(llm | StrOutputParser(), summary_prompt, message_placeholder)
                else:
                    chat_history = build_chat_history(current_messages[:-1])
                    retrieved_docs = qa_setup["retriever"].invoke({
                        "input": user_prompt,
                        "chat_history": chat_history,
                    })
                    context_str = "\n\n".join(f"[Chunk {i}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs, 1))
                    subject_prompt = selected_cfg["prompt"] + "\n\nProvide a concise bullet-point summary for this topic."
                    prompt = ChatPromptTemplate.from_template(subject_prompt)
                    formatted_prompt = prompt.format(context=context_str, input=user_prompt)
                    full_response = stream_with_retry(qa_setup["llm"] | StrOutputParser(), formatted_prompt, message_placeholder)
                    current_sources = extract_citation_metadata(retrieved_docs)
                    if current_sources:
                        with st.expander("📚 View Source Citations & Passages"):
                            for idx, src in enumerate(current_sources, 1):
                                st.markdown(f"**[{idx}] {src['source']}{src['page']}**")
                                st.caption(f"\"{src['snippet']}\"")

            # PATH C: Standard Academic Course Questions
            else:
                chat_history = build_chat_history(current_messages[:-1])
                query_for_retrieval = clean_query(user_prompt)
                if needs_reformulation(user_prompt, bool(chat_history)):
                    retrieved_docs = qa_setup["retriever"].invoke({
                        "input": query_for_retrieval,
                        "chat_history": chat_history,
                    })
                    low_confidence = False
                else:
                    scored_docs = vector_store.similarity_search_with_score(query_for_retrieval, k=5)
                    retrieved_docs = [doc for doc, _ in scored_docs]
                    top_score = scored_docs[0][1] if scored_docs else 0
                    low_confidence = top_score > 0.65
                context_str = "\n\n".join(f"[Chunk {i}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs, 1))
                formatted_prompt = qa_setup["prompt"].format(context=context_str, input=user_prompt)
                full_response = stream_with_retry(qa_setup["llm"] | StrOutputParser(), formatted_prompt, message_placeholder)
                if low_confidence:
                    full_response += "\n\n*⚠️ This answer may not be directly from your course notes — verify with your materials.*"
                    message_placeholder.markdown(full_response)
                current_sources = extract_citation_metadata(retrieved_docs)
                if current_sources:
                    with st.expander("📚 View Source Citations & Passages"):
                        for idx, src in enumerate(current_sources, 1):
                            st.markdown(f"**[{idx}] {src['source']}{src['page']}**")
                            st.caption(f"\"{src['snippet']}\"")
                # Follow-up suggestions
                followups = generate_followups(retrieved_docs)
                if followups:
                    st.markdown("<small style='color:#9CA3AF;'>💡 Suggested follow-ups:</small>", unsafe_allow_html=True)
                    cols = st.columns(len(followups))
                    for col, q in zip(cols, followups):
                        with col:
                            if st.button(q, key=f"fu_{hash(q)}", use_container_width=True):
                                st.session_state.followup_clicked = q
                # Flashcard button
                if st.button("🃏 Generate Flashcards", key=f"fc_{len(current_messages)}"):
                    st.session_state.generate_flashcards = True
                    st.session_state.flashcard_docs = retrieved_docs

            if not full_response:
                full_response = (
                    "I couldn't find relevant information in the course notes. "
                    "Please try rephrasing your question."
                )

        except Exception as exc:
            msg = str(exc).lower()
            if any(code in msg for code in ["429", "resource_exhausted", "quota", "rate_limit"]):
                full_response = (
                    "⚠️ **All models are currently busy.** Rate limits reached across all available models. "
                    "Please wait about 30 seconds before submitting your next question."
                )
            elif any(code in msg for code in ["503", "unavailable", "overloaded"]):
                full_response = (
                    "⚠️ **Server Busy.** Google Gemini is temporarily overloaded. "
                    "Please wait a moment and try asking your question again."
                )
            else:
                full_response = (
                    "⚠️ **System Interruption.** Unable to process your request at the moment. "
                    "If this issue persists, contact **Mwesigwa Mark** at **+256 701913028** for support."
                )

    assistant_msg = {"role": "assistant", "content": full_response}
    if current_sources:
        assistant_msg["sources"] = current_sources
    current_messages.append(assistant_msg)
    save_session_history(session_id, st.session_state.subject_messages)

# ── Flashcard Panel ───────────────────────────────────────────────────────────
if st.session_state.get("generate_flashcards") and st.session_state.get("flashcard_docs"):
    st.session_state.generate_flashcards = False
    with st.spinner("Generating flashcards…"):
        # gemini-1.5-flash — reliable structured JSON output
        cards = generate_flashcards(st.session_state.flashcard_docs, load_llm_gemini_fallback())
    if cards:
        st.markdown("---")
        st.markdown("#### 🃏 Flashcards")
        if "fc_scores" not in st.session_state:
            st.session_state.fc_scores = {}
        total_answered = 0
        score_sum = 0
        for i, card in enumerate(cards):
            with st.expander(f"Q{i+1}: {card['question']}"):
                st.markdown(f"**Answer:** {card['answer']}")
                ans_key = f"fc_ans_{i}_{len(current_messages)}"
                sub_key = f"fc_sub_{i}_{len(current_messages)}"
                user_ans = st.text_input("Your answer:", key=ans_key)
                if st.button("Submit", key=sub_key) and user_ans.strip():
                    grade_prompt = (
                        f"Question: {card['question']}\n"
                        f"Correct answer: {card['answer']}\n"
                        f"Student answer: {user_ans}\n\n"
                        "Score the student's answer from 0 to 100 based on accuracy, completeness, "
                        'and use of key terms. Reply ONLY with valid JSON: {"score": <number>, "feedback": "<one sentence>"}'
                    )
                    try:
                        # qwen3.8-27b — fast grading on Groq inference chips
                        grader = load_groq_llm_primary() or load_llm()
                        raw = (grader | StrOutputParser()).invoke(grade_prompt)
                        result = _parse_score_json(raw)
                        sc = int(result.get("score", 0))
                        fb = result.get("feedback", "")
                        st.session_state.fc_scores[i] = sc
                        st.markdown(f"{score_badge(sc)}  {fb}")
                    except Exception as e:
                        st.warning("⚠️ Could not grade your answer. Please contact **Mwesigwa Mark** at **+256 701913028** for support.")
                    total_answered += 1
                    score_sum += st.session_state.fc_scores[i]
        if total_answered > 0:
            avg = score_sum // total_answered
            st.markdown(f"---\n**{total_answered}/{len(cards)} answered — Average: {avg}/100**")
    else:
        st.warning("Could not generate flashcards for this answer.")

# ── Quiz Mode (MCQ) ──────────────────────────────────────────────────────────
if st.session_state.get("quiz_mode"):
    if "mcq_score" not in st.session_state:
        st.session_state.mcq_score = {"total": 0, "sum": 0}

    st.markdown("---")
    st.markdown("#### 📝 Quiz Mode — Multiple Choice")

    # Generate a new MCQ if none active — gemini-1.5-flash (reliable JSON structure)
    if not st.session_state.get("mcq_question"):
        with st.spinner("Generating question…"):
            try:
                mcq = generate_mcq(vector_store, load_llm_gemini_fallback())
            except Exception as e:
                st.warning("⚠️ Could not generate a question. Please contact **Mwesigwa Mark** at **+256 701913028** for support.")
                mcq = {}
        if mcq:
            st.session_state.mcq_question = mcq
            st.session_state.mcq_answered = False
        else:
            st.warning("⚠️ Could not generate a question. Please contact **Mwesigwa Mark** at **+256 701913028** for support.")

    mcq = st.session_state.get("mcq_question")
    if mcq:
        st.markdown(f"**{mcq['question']}**")
        option_labels = [f"{k}: {v}" for k, v in mcq["options"].items()]

        if not st.session_state.get("mcq_answered"):
            selected = st.radio("Choose your answer:", option_labels, key="mcq_radio", index=None)
            if st.button("✅ Submit", key="mcq_submit"):
                if not selected:
                    st.warning("Please select an option first.")
                else:
                    chosen_letter = selected.split(":")[0].strip()
                    correct_letter = mcq["answer"].strip().upper()
                    correct_text = mcq["options"].get(correct_letter, "")

                    # Ask Gemini to score with partial credit reasoning
                    grade_prompt = (
                        f"Question: {mcq['question']}\n"
                        f"Options: {json.dumps(mcq['options'])}\n"
                        f"Correct answer: {correct_letter} - {correct_text}\n"
                        f"Student selected: {chosen_letter} - {mcq['options'].get(chosen_letter, '')}\n"
                        f"Explanation: {mcq['explanation']}\n\n"
                        "Award a score from 0 to 100. If the student chose the correct answer give 100. "
                        "If wrong, reason about how close or related their choice is to the correct answer "
                        "and award partial marks accordingly. "
                        'Reply ONLY with valid JSON: {"score": <number>, "feedback": "<one sentence>"}'
                    )
                    try:
                        # qwen3.8-27b — fast grading on Groq inference chips
                        grader = load_groq_llm_primary() or load_llm()
                        raw = (grader | StrOutputParser()).invoke(grade_prompt)
                        result = _parse_score_json(raw)
                        sc = int(result.get("score", 0))
                        fb = result.get("feedback", "")
                        st.session_state.mcq_last_score = sc
                        st.session_state.mcq_last_feedback = fb
                        st.session_state.mcq_last_chosen = chosen_letter
                        st.session_state.mcq_answered = True
                        qs = st.session_state.mcq_score
                        qs["total"] += 1
                        qs["sum"] += sc
                        st.session_state.mcq_score = qs
                        st.rerun()
                    except Exception as e:
                        st.warning("⚠️ Could not grade your answer. Please contact **Mwesigwa Mark** at **+256 701913028** for support.")
            # Show result after answering
            correct_letter = mcq["answer"].strip().upper()
            chosen_letter = st.session_state.get("mcq_last_chosen", "")
            sc = st.session_state.get("mcq_last_score", 0)
            fb = st.session_state.get("mcq_last_feedback", "")

            for k, v in mcq["options"].items():
                if k == correct_letter and k == chosen_letter:
                    st.markdown(f"**{k}: {v}** ✅")
                elif k == correct_letter:
                    st.markdown(f"**{k}: {v}** ← Correct answer")
                elif k == chosen_letter:
                    st.markdown(f"~~{k}: {v}~~ ← Your answer")
                else:
                    st.markdown(f"{k}: {v}")

            st.markdown(f"{score_badge(sc)}")
            st.markdown(f"*{fb}*")
            st.caption(f"💡 {mcq['explanation']}")

            if st.button("Next Question ➡️", key="mcq_next"):
                st.session_state.pop("mcq_question", None)
                st.session_state.pop("mcq_answered", None)
                st.session_state.pop("mcq_last_score", None)
                st.session_state.pop("mcq_last_feedback", None)
                st.session_state.pop("mcq_last_chosen", None)
                st.rerun()

# ── Test Me Mode — Question & Evaluation ─────────────────────────────────────
if st.session_state.get("test_me_mode"):
    if "test_me_score" not in st.session_state:
        st.session_state.test_me_score = {"correct": 0, "total": 0, "sum": 0}

    st.markdown("---")
    st.markdown("#### 🧠 Test Me Mode")

    # Generate a new question if none is active — compound-mini (fast, short output)
    if not st.session_state.get("test_me_question"):
        seed_terms = ["definition", "concept", "theory", "formula", "method", "model", "analysis"]
        import random
        seed = random.choice(seed_terms)
        test_docs = vector_store.similarity_search(seed, k=3)
        if test_docs:
            chunk = random.choice(test_docs).page_content[:600]
            q_prompt = (
                f"From the following course content, write ONE clear exam-style question "
                f"a student should be able to answer. Return ONLY the question, nothing else.\n\n{chunk}"
            )
            try:
                q_llm = load_groq_llm_secondary() or load_llm()
                question = (q_llm | StrOutputParser()).invoke(q_prompt).strip()
                st.session_state.test_me_question = question
                st.session_state.test_me_context = chunk
            except Exception as e:
                st.warning("⚠️ Could not generate a question. Please contact **Mwesigwa Mark** at **+256 701913028** for support.")

    if st.session_state.get("test_me_question"):
        st.markdown(f"**Question:** {st.session_state.test_me_question}")
        student_ans = st.text_area("Your answer:", key="test_me_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Submit Answer", key="test_me_submit") and student_ans.strip():
                eval_prompt = (
                    f"Course content: {st.session_state.test_me_context}\n"
                    f"Question: {st.session_state.test_me_question}\n"
                    f"Student answer: {student_ans}\n\n"
                    "Score the student's answer from 0 to 100 based on accuracy, completeness, "
                    'and use of key terms. Reply ONLY with valid JSON: {"score": <number>, "feedback": "<two sentences max>"}'
                )
                try:
                    # qwen3.8-27b — fast grading on Groq inference chips
                    grader = load_groq_llm_primary() or load_llm()
                    raw = (grader | StrOutputParser()).invoke(eval_prompt)
                    result = _parse_score_json(raw)
                    sc = int(result.get("score", 0))
                    fb = result.get("feedback", "")
                    st.markdown(f"{score_badge(sc)}")
                    st.markdown(f"*{fb}*")
                    s = st.session_state.test_me_score
                    s["total"] += 1
                    s["sum"] += sc
                    if sc >= 75:
                        s["correct"] += 1
                    st.session_state.test_me_score = s
                    st.session_state.pop("test_me_question", None)
                    st.session_state.pop("test_me_context", None)
                except Exception as e:
                        st.warning("⚠️ Could not evaluate your answer. Please contact **Mwesigwa Mark** at **+256 701913028** for support.")
        with col2:
            if st.button("⏭️ Skip Question", key="test_me_skip"):
                st.session_state.pop("test_me_question", None)
                st.session_state.pop("test_me_context", None)
                st.rerun()

# ── Follow-up JS Injector — paste question into native chat input ─────────────
if st.session_state.get("followup_clicked"):
    q = st.session_state.pop("followup_clicked")
    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const textarea = doc.querySelector('[data-testid="stChatInput"] textarea');
            if (!textarea) return;
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
            nativeInputValueSetter.call(textarea, {json.dumps(q)});
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.focus();
        }})();
        </script>
        """,
        height=0,
        width=0,
    )

# ── Dynamic Copy Button Injector (DOM Component) ──────────────────────────────
components.html(
    """
    <script>
    function injectCopyButtons() {
        const doc = window.parent.document;
        const messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
        
        messages.forEach((msg) => {
            const isAssistant = msg.querySelector('[data-testid="stChatMessageAvatarAssistant"]');
            const alreadyHasBtn = msg.querySelector('.bsqe2-copy-btn');
            
            if (isAssistant && !alreadyHasBtn) {
                const btn = doc.createElement('button');
                btn.className = 'bsqe2-copy-btn';
                btn.innerHTML = '📋 Copy Answer';
                btn.style.cssText = `
                    background: transparent;
                    border: 1px solid rgba(156, 163, 175, 0.35);
                    color: inherit;
                    opacity: 0.85;
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-size: 11px;
                    font-weight: 500;
                    cursor: pointer;
                    margin-top: 8px;
                    margin-bottom: 4px;
                    transition: all 0.2s ease;
                `;
                
                btn.onmouseover = () => { btn.style.opacity = '1'; btn.style.borderColor = '#2563EB'; };
                btn.onmouseout = () => { btn.style.opacity = '0.85'; btn.style.borderColor = 'rgba(156, 163, 175, 0.35)'; };
                
                btn.onclick = () => {
                    const markdownEl = msg.querySelector('[data-testid="stMarkdownContainer"]');
                    if (!markdownEl) return;
                    const textToCopy = markdownEl.innerText;

                    function doFallback(text) {
                        try {
                            const pDoc = window.parent.document;
                            const textarea = pDoc.createElement('textarea');
                            textarea.value = text;
                            textarea.style.position = 'fixed';
                            textarea.style.left = '-9999px';
                            textarea.style.top = '-9999px';
                            pDoc.body.appendChild(textarea);
                            textarea.focus();
                            textarea.select();
                            const success = pDoc.execCommand('copy');
                            pDoc.body.removeChild(textarea);
                            if (success) {
                                btn.innerHTML = '✅ Copied!';
                                setTimeout(() => { btn.innerHTML = '📋 Copy Answer'; }, 1800);
                            } else {
                                btn.innerHTML = '❌ Failed';
                                setTimeout(() => { btn.innerHTML = '📋 Copy Answer'; }, 1800);
                            }
                        } catch (err) {
                            console.error('Copy fallback failed:', err);
                            btn.innerHTML = '❌ Error';
                            setTimeout(() => { btn.innerHTML = '📋 Copy Answer'; }, 1800);
                        }
                    }

                    if (window.parent && window.parent.navigator && window.parent.navigator.clipboard && window.parent.navigator.clipboard.writeText) {
                        window.parent.navigator.clipboard.writeText(textToCopy)
                            .then(() => {
                                btn.innerHTML = '✅ Copied!';
                                setTimeout(() => { btn.innerHTML = '📋 Copy Answer'; }, 1800);
                            })
                            .catch(() => {
                                doFallback(textToCopy);
                            });
                    } else {
                        doFallback(textToCopy);
                    }
                };
                
                msg.appendChild(btn);
            }
        });
    }
    setInterval(injectCopyButtons, 600);
    </script>
    """,
    height=0,
    width=0,
)