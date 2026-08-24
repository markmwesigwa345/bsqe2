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

import os
import re
import streamlit as st
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
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    st.error(f"❌ Missing dependency: **{e}**")
    st.info(
        "Run `pip install -r requirements.txt` to install all required packages, "
        "then restart the app."
    )
    st.stop()

# ── Subject Registry ──────────────────────────────────────────────────────────
from subjects_config import SUBJECTS

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Environment / API Key ─────────────────────────────────────────────────────
# Priority: Streamlit Cloud secrets → .env file → environment variable
load_dotenv()
try:
    if "GOOGLE_API_KEY" in st.secrets:
        # Strip potential whitespace or quotes around the key from secrets
        os.environ["GOOGLE_API_KEY"] = str(st.secrets["GOOGLE_API_KEY"]).strip().strip('"').strip("'")
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
    header[data-testid="stHeader"] button, 
    header[data-testid="stHeader"] svg {{
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

    /* Spinners */
    [data-testid="stSpinner"] p {{
        color: var(--text-main) !important;
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
    [data-testid="stChatMessage"] span {{
        color: var(--text-main) !important;
    }}
    [data-testid="stChatMessage"] code {{
        background-color: var(--input-bg) !important;
        color: var(--text-main) !important;
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

# ── Session State — Subject Switching ────────────────────────────────────────
if "active_subject" not in st.session_state:
    st.session_state.active_subject = selected_subject_name
    st.session_state.messages = []

if st.session_state.active_subject != selected_subject_name:
    st.session_state.active_subject = selected_subject_name
    st.session_state.messages = []
    st.rerun()

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
    """Load and cache the Gemini LLM (built once, reused on every message)."""
    return ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


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
        else os.path.join(SCRIPT_DIR, faiss_dir)
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
def setup_qa_chain(_vector_store, subject_prompt: str):
    """Build and cache the LCEL QA chain for the given vector store and prompt.

    Returns a flat chain: {"input": str} → str (answer tokens streamed).
    Uses StrOutputParser so .stream() yields plain strings directly.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.6-flash", google_api_key=api_key)
    prompt = ChatPromptTemplate.from_template(subject_prompt)
    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_for_llm(inputs: dict) -> dict:
        return {
            "context": format_docs(inputs["context"]),
            "input": inputs["input"],
        }

    setup_and_retrieval = RunnableParallel(
        {
            "context": lambda x: retriever.invoke(x["input"]),
            "input": lambda x: x["input"],
        }
    )

    # Flat chain: retrieval → format → prompt → LLM → plain string
    qa_chain = setup_and_retrieval | format_for_llm | prompt | llm | StrOutputParser()

    return qa_chain


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
    qa_chain = setup_qa_chain(vector_store, selected_cfg["prompt"])
except Exception as exc:
    st.error(f"❌ Failed to set up the AI chain: {exc}")
    st.stop()

# Welcome message (shown only when the chat is empty)
if not st.session_state.messages:
    st.success(
        f"Hi there 📊 I am **BSQE AI**, your study assistant for Bachelor of Science in Quantitative Economics. "
        f"Ready to assist with **{selected_subject_name}**. Ask any question below!"
    )

# ── Chat History ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input & Response ─────────────────────────────────────────────────────
if user_prompt := st.chat_input(f"Ask a question about {selected_subject_name}…"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""  # Ensure full_response is always defined (BUG-3 fix)

        try:
            # PATH A: Casual Greetings / Pleasantries (Bypasses FAISS Retrieval completely)
            if is_conversational_query(user_prompt):
                llm = load_llm()
                greeting_prompt = (
                    f"You are BSQE2 AI, an elite study assistant for Bachelor of Science in Quantitative Economics students "
                    f"currently helping with {selected_subject_name}.\n\n"
                    f"Respond warmly, naturally, and concisely to this user greeting: '{user_prompt}'."
                )
                response_stream = (llm | StrOutputParser()).stream(greeting_prompt)
                full_response = st.write_stream(response_stream)

            # PATH B: Explicit User Summarization Request (Summarizes previous answer if available)
            elif is_summarization_request(user_prompt):
                llm = load_llm()

                # Find the last assistant message in chat history
                last_assistant_msg = None
                for msg in reversed(st.session_state.messages[:-1]):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg["content"]
                        break

                if last_assistant_msg:
                    summary_prompt = (
                        f"You are BSQE2 AI. Summarize the following answer clearly into concise bullet points, "
                        f"highlighting key definitions and main takeaways:\n\n{last_assistant_msg}"
                    )
                    response_stream = (llm | StrOutputParser()).stream(summary_prompt)
                    full_response = st.write_stream(response_stream)
                else:
                    # Fallback if no prior answer exists: treat as standard RAG query with summary instruction
                    with st.spinner("Thinking…"):
                        subject_prompt = (
                            selected_cfg["prompt"]
                            + "\n\nProvide a concise bullet-point summary for this topic."
                        )
                        prompt = ChatPromptTemplate.from_template(subject_prompt)
                        retriever = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5},
                        )
                        docs = retriever.invoke(user_prompt)
                        context_str = "\n\n".join(doc.page_content for doc in docs)
                        formatted_prompt = prompt.format(context=context_str, input=user_prompt)
                        response_stream = (llm | StrOutputParser()).stream(formatted_prompt)
                        full_response = st.write_stream(response_stream)

            # PATH C: Standard Academic Course Questions (Full, Detailed, Step-by-Step Response by default)
            else:
                with st.spinner("Thinking…"):
                    # Use the cached qa_chain — no need to rebuild LLM/retriever/prompt per message
                    response_stream = qa_chain.stream({"input": user_prompt})
                    full_response = st.write_stream(response_stream)

            if not full_response:
                full_response = (
                    "I couldn't find relevant information in the course notes. "
                    "Please try rephrasing your question."
                )

        except Exception as exc:
            full_response = (
                f"❌ An error occurred: `{exc}`\n\nPlease try again later."
            )
            st.error(f"Error type: **{type(exc).__name__}**")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})