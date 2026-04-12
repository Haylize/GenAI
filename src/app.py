import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from router.router import AssistantRouter

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="",
    layout="centered",
)

# --- GLOBAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #1a3c34 !important;
    }

    .stApp {
        background-color: #f7f5f2;
    }

    /* Force all headings to dark green */
    h1, h2, h3, h4, h5, h6 {
        color: #1a3c34 !important;
        font-family: 'DM Serif Display', serif !important;
    }

    /* Force all text to dark green */
    p, span, div, li {
        color: #1a3c34;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: white !important;
        border-radius: 12px !important;
        padding: 8px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div {
        color: #222 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2d6a4f !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        transition: background 0.2s !important;
    }

    .stButton > button:hover {
        background-color: #1a3c34 !important;
        color: white !important;
    }

    /* Route tags */
    .route-tag {
        display: inline-block;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .route-rag { background: #e8f5e9; color: #1a3c34 !important; border: 1px solid #a5d6a7; }
    .route-calculator { background: #e3f2fd; color: #0d47a1 !important; border: 1px solid #90caf9; }
    .route-weather { background: #fff3e0; color: #e65100 !important; border: 1px solid #ffcc80; }
    .route-web_search { background: #f3e5f5; color: #4a148c !important; border: 1px solid #ce93d8; }
    .route-chat { background: #f5f5f5; color: #333 !important; border: 1px solid #ccc; }

    /* Sources */
    .source-tag {
        display: inline-block;
        background: #e8f5e9;
        color: #1a3c34 !important;
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
        color: #888 !important;
        margin-bottom: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Home cards */
    .home-card {
        background: white;
        border-radius: 12px;
        padding: 20px 22px;
        border-left: 4px solid #2d6a4f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .home-card h4 {
        margin: 0 0 8px 0;
        color: #1a3c34 !important;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .home-card p {
        margin: 0;
        color: #555 !important;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .home-disclaimer {
        background: #fff8e1;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #f59e0b;
        font-size: 0.85rem;
        color: #78350f !important;
        line-height: 1.6;
        margin-top: 24px;
    }

    .home-disclaimer strong {
        color: #78350f !important;
    }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE INIT ---
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "router" not in st.session_state:
    st.session_state.router = None


# --- LOAD ROUTER (cached) ---
@st.cache_resource
def load_router():
    return AssistantRouter()


ROUTE_LABELS = {
    "rag": "Medical knowledge base",
    "calculator": "Calculator",
    "weather": "Weather",
    "web_search": "Web search",
    "chat": "General conversation",
}


# --- HOME PAGE ---
def render_home():
    # Hero block
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a3c34 0%, #2d6a4f 100%);
                border-radius: 16px; padding: 48px 40px; margin-bottom: 24px;">
        <div style="font-family: 'DM Serif Display', serif; font-size: 2.6rem;
                    color: white; margin-bottom: 12px;">Medical Assistant</div>
        <div style="font-size: 1.05rem; color: rgba(255,255,255,0.85); line-height: 1.7;">
            An AI-powered assistant to help you understand symptoms, conditions,
            and when to seek medical care — based on verified medical knowledge.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Start chatting button — centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start chatting", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    # Section title
    st.markdown("""
    <div style="font-family: 'DM Serif Display', serif; font-size: 1.3rem;
                color: #1a3c34; margin: 24px 0 16px 0;">How it works</div>
    """, unsafe_allow_html=True)

    # Info cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="home-card">
            <h4>Ask a question</h4>
            <p>Describe your symptoms or ask about a condition in plain language, in any language.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="home-card">
            <h4>Smart routing</h4>
            <p>The assistant detects whether to search the medical knowledge base, calculate, check weather, or just chat.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="home-card">
            <h4>Conversation memory</h4>
            <p>The assistant remembers the last 5 exchanges so you can ask follow-up questions naturally.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="home-card">
            <h4>Local and private</h4>
            <p>Everything runs locally on your machine. No data is sent to external servers.</p>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="home-disclaimer">
        <strong>Important disclaimer</strong><br>
        This assistant is for informational purposes only and does not replace professional medical advice.
        Always consult a qualified healthcare provider for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)


# --- CHAT PAGE ---
def render_chat():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown('<h2 style="color:#1a3c34; font-family: DM Serif Display, serif;">Medical Assistant</h2>', unsafe_allow_html=True)
    with col2:
        if st.button("Home"):
            st.session_state.page = "home"
            st.rerun()

    # Load router
    if st.session_state.router is None:
        with st.spinner("Loading assistant..."):
            try:
                st.session_state.router = load_router()
            except Exception as e:
                st.error(f"Failed to load the assistant: {e}")
                st.stop()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("route"):
                route = message["route"]
                label = ROUTE_LABELS.get(route, route)
                st.markdown(
                    f'<span class="route-tag route-{route}">{label}</span>',
                    unsafe_allow_html=True
                )
            st.markdown(f'<p style="color:#222; margin:0;">{message["content"]}</p>', unsafe_allow_html=True)

            if message["role"] == "assistant" and message.get("sources"):
                st.markdown('<div class="sources-block"><div class="sources-label">Sources</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="source-tag">{s}</span>' for s in message["sources"]])
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Clear conversation button
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.router.memory.clear()
            st.rerun()

    # Chat input
    user_input = st.chat_input("Ask your question...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(f'<p style="color:#222; margin:0;">{user_input}</p>', unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "route": None,
            "sources": [],
        })

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = st.session_state.router.route(user_input)

            route = result.get("type", "chat")
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            label = ROUTE_LABELS.get(route, route)
            st.markdown(
                f'<span class="route-tag route-{route}">{label}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f'<p style="color:#222; margin:0;">{answer}</p>', unsafe_allow_html=True)

            if sources:
                st.markdown('<div class="sources-block"><div class="sources-label">Sources</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="source-tag">{s}</span>' for s in sources])
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "route": route,
            "sources": sources,
        })


# --- ROUTER ---
if st.session_state.page == "home":
    render_home()
else:
    render_chat()