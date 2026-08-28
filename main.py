"""
main.py — BSQE2 AI (Single-Subject Legacy App)
─────────────────────────────────────────────────────────────────────────────
NOTE: This is the older single-subject version of the app.
      The active production entry point is ibt.py (multi-subject).
      This file is kept for reference but should NOT be used as the
      Streamlit Cloud main file.
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Updated imports — langchain_huggingface replaces deprecated langchain_community embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings          # ← fixed
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# MUST be the very first Streamlit call
st.set_page_config(
    page_title="BSQE2 AI 🎓",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

load_dotenv()
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CONFIGURE THIS BEFORE RUNNING main.py ─────────────────────────────────────
# Set VECTOR_STORE_DIR to the name of the FAISS folder you generated via
# colab.ipynb and placed in the same directory as this file.
# Example: "micro_economics_faiss"  or  "financial_analysis_faiss"
# NOTE: This file is the legacy single-subject app. The active multi-subject
#       production app is ibt.py — use that instead.
# ──────────────────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = os.path.join(SCRIPT_DIR, "YOUR_FAISS_FOLDER_HERE")

# Sidebar contents
with st.sidebar:
    st.title("BSQE2 Assistant 📊")
    st.markdown('''
        ## About
        This app was designed for BSQE2 students to ease their revision process.
    ''')
    st.write('Made by Mwesigwa Mark')

    if os.path.isdir(VECTOR_STORE_DIR):
        st.info("📊 Vector DB: loaded")


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner=False)
def load_vector_store(folder_path: str):
    """Load the pre-created FAISS vector store from its native folder."""
    if not os.path.isdir(folder_path):
        st.error(f"❌ Vector store folder not found: `{folder_path}`")
        st.info("💡 Run the Colab notebook to generate the FAISS folder, then place it here.")
        st.stop()

    try:
        embeddings = load_embeddings()
        vector_store = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        st.success("✅ Vector store loaded successfully!")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error loading vector store: {e}")
        st.info("💡 Re-generate the FAISS folder using the updated colab.ipynb notebook.")
        st.stop()


@st.cache_resource(show_spinner=False)
def setup_qa_chain(_vector_store):
    """Set up the QA chain using modern LangChain LCEL approach."""
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")

    prompt = ChatPromptTemplate.from_template("""
    You are an expert in Micro Economics for Bachelor of Science in Quantitative Economics (BSQE2) students.
    Answer the following question based on the provided context.
    Think step by step and provide a clear, detailed answer using formal economic terms and mathematical derivations where appropriate.
    If the context doesn't contain enough information, say so but go ahead and use your general economic knowledge to answer.

    <context>
    {context}
    </context>

    Question: {input}

    Answer:""")

    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_for_llm(inputs):
        return {
            "context": format_docs(inputs["context"]),
            "input": inputs["input"],
        }

    setup_and_retrieval = RunnableParallel(
        {"context": lambda x: retriever.invoke(x["input"]), "input": lambda x: x["input"]}
    )

    answer_chain = format_for_llm | prompt | llm | StrOutputParser()

    qa_chain = setup_and_retrieval | RunnableParallel(
        {"answer": answer_chain, "context": lambda x: x["context"]}
    )

    return qa_chain


def main():
    st.header("🎓 BSQE2 AI - Micro Economics Assistant")

    try:
        with st.spinner("Loading embeddings…"):
            embeddings = load_embeddings()

        with st.spinner("Loading vector store — please wait…"):
            vector_store = load_vector_store(VECTOR_STORE_DIR)

        with st.spinner("Setting up AI chain…"):
            qa_chain = setup_qa_chain(vector_store)

    except Exception as e:
        st.error(f"❌ Failed to load resources: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask questions…."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = qa_chain.invoke({"input": prompt})
                    answer = response.get("answer", "").strip()

                    if not answer:
                        answer = "I couldn't find relevant information. Please try rephrasing your question."

                    st.markdown(answer)



                except Exception as e:
                    answer = f"❌ An error occurred: `{e}`\n\nPlease try again."
                    st.error(f"Error type: **{type(e).__name__}**")
                    st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
