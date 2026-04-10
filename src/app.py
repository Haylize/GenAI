"""
Streamlit application for the Medical Assistant.

Two pages:
- Home: presentation and usage instructions
- Chat: interactive RAG-powered medical assistant with memory
"""

import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectorstore import load_local_retriever
from qa_chain import get_medical_rag_chain
from memory import ConversationMemory

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="",
    layout="centered",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    .main {
        background-color: #f7f5f2;
    }

    .stApp {
        background-color: #f7f5f2;
    }

    .home-hero {
        background: linear-gradient(135deg, #1a3c34 0%, #2d6a4f 100%);
        border-radius: 16px;
        padding: 48px 40px;
        color: white;
        margin-bottom: 32px;
    }

    .home-hero h1 {
        font-size: 2.6rem;
        margin-bottom: 12px;
        color: white;
    }

    .home-hero p {
        font-size: 1.1rem;
        opacity: 0.85;
        line-height: 1.7;
    }

    .info-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        border-left: 4px solid #2d6a4f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .info-card h3 {
        margin-top: 0;
        color: #1a3c34;
        font-size: 1.1rem;
    }

    .info-card p {
        margin: 0;
        color: #555;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .warning-card {
        background: #fff8e1;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid #f59e0b;
        margin-top: 24px;
        font-size: 0.9rem;
        color: #78350f;
        line-height: 1.6;
    }

    .source-tag {
        display: inline-block;
        background: #e8f5e9;
        color: #1a3c34;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 4px 4px 0 0;
        border: 1px solid #a5d6a7;
    }

    .sources-block {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
    }

    .sources-label {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stChatMessage"] {
        background: white;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .stButton > button {
        background-color: #2d6a4f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background-color: #1a3c34;
        color: white;
    }

    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ccc;
        font-family: 'DM Sans', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE INIT ---
if "page" not in st.session_state:
    st.session_state.page = "home"

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_exchanges=5)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "sources"}

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# --- LOAD RAG SYSTEM (cached) ---
@st.cache_resource
def load_rag_system():
    retriever = load_local_retriever()
    chain = get_medical_rag_chain(retriever)
    return retriever, chain


# --- HOME PAGE ---
def render_home():
    st.markdown("""
    <div class="home-hero">
        <h1>Medical Assistant</h1>
        <p>An AI-powered assistant to help you understand symptoms, conditions,
        and when to seek medical care — based on verified medical knowledge.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How it works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Ask a question</h3>
            <p>Describe your symptoms or ask about a condition in plain language, in any language.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>Get sourced answers</h3>
            <p>Each response indicates which medical files were used to generate the answer.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Conversation memory</h3>
            <p>The assistant remembers the last 5 exchanges so you can ask follow-up questions naturally.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>Local and private</h3>
            <p>Everything runs locally on your machine. No data is sent to external servers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-card">
        <strong>Important disclaimer</strong><br>
        This assistant is for informational purposes only and does not replace professional medical advice.
        Always consult a qualified healthcare provider for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Start chatting"):
        st.session_state.page = "chat"
        st.rerun()


# --- CHAT PAGE ---
def render_chat():
    st.markdown("## Medical Assistant")

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Home"):
            st.session_state.page = "home"
            st.rerun()

    # Load RAG system
    if st.session_state.rag_chain is None:
        with st.spinner("Loading medical knowledge base..."):
            try:
                retriever, chain = load_rag_system()
                st.session_state.retriever = retriever
                st.session_state.rag_chain = chain
            except Exception as e:
                st.error(f"Failed to load the system: {e}")
                st.stop()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown('<div class="sources-block"><div class="sources-label">Sources</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="source-tag">{s}</span>' for s in message["sources"]])
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Clear memory button
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()

    # Chat input
    user_input = st.chat_input("Ask your question...")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "sources": [],
        })

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                chain_input = {
                    "question": user_input,
                    "history": st.session_state.memory.get_langchain_messages(),
                }
                answer = st.session_state.rag_chain.invoke(chain_input)
                docs = st.session_state.retriever.invoke(user_input)

                # Extract unique sources
                seen = set()
                sources = []
                for doc in docs:
                    title = doc.metadata.get("disease_title", "Unknown")
                    if title not in seen:
                        sources.append(title)
                        seen.add(title)

            st.write(answer)

            if sources:
                st.markdown('<div class="sources-block"><div class="sources-label">Sources</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="source-tag">{s}</span>' for s in sources])
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Save to history and memory
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.session_state.memory.add_exchange(user_input, answer)


# --- ROUTER ---
if st.session_state.page == "home":
    render_home()
else:
    render_chat()

