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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    body {
        color: #FAFAFA;
        background-color: #020203;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .subject-badge {
        display: inline-block;
        background: #007bff;
        color: white;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("BSQE2 AI Assistant 📊")
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
    """
    cleaned = text.strip().lower()

    # Never treat summarization requests as casual greetings
    if any(kw in cleaned for kw in ["summarize", "summary", "brief", "recap", "short version", "bullet points"]):
        return False
    
    # Exact match for common short greetings / pleasantries
    casual_phrases = {
        "hi", "hello", "hey", "hey there", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "thanks", "thank you",
        "who created you", "who made you", "what can you do", "bye", "goodbye"
    }
    if cleaned in casual_phrases:
        return True

    # Pattern matching for greetings with punctuation or extra words
    greeting_patterns = [
        r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)[\!\?\.,\s]*",
        r"^(thank you|thanks|bye|goodbye)[\!\?\.,\s]*",
        r"^(who (created|made|built) (you|this app))",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, cleaned):
            # If the user asks a detailed question (>5 words), send it to retrieval
            if len(cleaned.split()) > 5:
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
    """Build the LCEL QA chain for the given vector store and prompt."""
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3-flash-preview", google_api_key=api_key)
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

    answer_chain = format_for_llm | prompt | llm | StrOutputParser()

    qa_chain = setup_and_retrieval | RunnableParallel(
        {"answer": answer_chain, "context": lambda x: x["context"]}
    )

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

        try:
            # PATH A: Casual Greetings / Pleasantries (Bypasses FAISS Retrieval completely)
            if is_conversational_query(user_prompt):
                api_key = os.getenv("GOOGLE_API_KEY")
                llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3-flash-preview", google_api_key=api_key)
                greeting_prompt = (
                    f"You are BSQE2 AI, an elite study assistant for Bachelor of Science in Quantitative Economics students "
                    f"currently helping with {selected_subject_name}.\n\n"
                    f"Respond warmly, naturally, and concisely to this user greeting: '{user_prompt}'."
                )
                response_stream = llm.stream(greeting_prompt)
                full_response = st.write_stream(response_stream)

            # PATH B: Explicit User Summarization Request (Summarizes previous answer if available)
            elif is_summarization_request(user_prompt):
                api_key = os.getenv("GOOGLE_API_KEY")
                llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3-flash-preview", google_api_key=api_key)

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
                    response_stream = llm.stream(summary_prompt)
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
                        response_stream = llm.stream(formatted_prompt)
                        full_response = st.write_stream(response_stream)

            # PATH C: Standard Academic Course Questions (Full, Detailed, Step-by-Step Response by default)
            else:
                with st.spinner("Thinking…"):
                    api_key = os.getenv("GOOGLE_API_KEY")
                    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3-flash-preview", google_api_key=api_key)
                    prompt = ChatPromptTemplate.from_template(selected_cfg["prompt"])
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5},
                    )

                    docs = retriever.invoke(user_prompt)
                    context_str = "\n\n".join(doc.page_content for doc in docs)

                    formatted_prompt = prompt.format(context=context_str, input=user_prompt)
                    response_stream = llm.stream(formatted_prompt)
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
