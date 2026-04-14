import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from router.router import AssistantRouter

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medical Assistant",
    layout="centered",
)

# --- GLOBAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --green-dark:   #1a3c34;
        --green-mid:    #2d6a4f;
        --green-light:  #a5d6a7;
        --cream:        #f7f5f2;
        --white:        #ffffff;
        --text-primary: #1a3c34;
        --text-body:    #2e2e2e;
        --text-muted:   #7a7a7a;
        --border:       #e4e1dc;
        --shadow-sm:    0 1px 4px rgba(0,0,0,0.07);
        --shadow-md:    0 4px 16px rgba(0,0,0,0.08);
        --radius-sm:    8px;
        --radius-md:    14px;
        --radius-lg:    20px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--cream);
    }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'DM Serif Display', serif !important;
        color: var(--text-primary) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }

    /* ── Chat messages ── */
    div[data-testid="stChatMessage"] {
        background: var(--white) !important;
        border-radius: var(--radius-md) !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--border) !important;
    }

    /* Fix text overflow */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div {
        color: var(--text-body) !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
        max-width: 100% !important;
    }

    /* User message — subtle tint */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #eef4f1 !important;
        border-color: #d0e8dc !important;
    }

    /* ── Chat input area ── */
    div[data-testid="stChatInputContainer"] {
        background: var(--white) !important;
        border-top: 1px solid var(--border) !important;
        padding: 12px 16px 14px !important;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.06) !important;
    }

    div[data-testid="stChatInputContainer"] textarea {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        color: var(--text-body) !important;
        border-radius: var(--radius-sm) !important;
        border: 1.5px solid var(--green-light) !important;
        background: var(--cream) !important;
        padding: 12px 16px !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: border-color 0.2s !important;
    }

    div[data-testid="stChatInputContainer"] textarea:focus {
        border-color: var(--green-mid) !important;
        outline: none !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.04), 0 0 0 3px rgba(45,106,79,0.1) !important;
    }

    div[data-testid="stChatInputContainer"] button {
        background: var(--green-mid) !important;
        border-radius: var(--radius-sm) !important;
        color: white !important;
        border: none !important;
        transition: background 0.2s !important;
    }

    div[data-testid="stChatInputContainer"] button:hover {
        background: var(--green-dark) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: var(--green-mid) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 22px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        white-space: nowrap !important;
        transition: background 0.2s, transform 0.1s !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stButton > button:hover {
        background-color: var(--green-dark) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Route tags ── */
    .route-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .route-rag         { background: #e8f5e9; color: #1a3c34 !important; border: 1px solid #a5d6a7; }
    .route-calculator  { background: #e3f2fd; color: #0d47a1 !important; border: 1px solid #90caf9; }
    .route-weather     { background: #fff3e0; color: #e65100 !important; border: 1px solid #ffcc80; }
    .route-web_search  { background: #f3e5f5; color: #4a148c !important; border: 1px solid #ce93d8; }
    .route-chat        { background: #f5f5f5; color: #333   !important; border: 1px solid #ddd; }

    /* ── Sources block ── */
    .sources-block {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--border);
    }

    .sources-label {
        font-size: 0.73rem;
        color: var(--text-muted) !important;
        margin-bottom: 6px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .source-tag {
        display: inline-block;
        background: #e8f5e9;
        color: var(--green-dark) !important;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 3px 3px 0 0;
        border: 1px solid var(--green-light);
    }

    /* ── Home cards ── */
    .home-card {
        background: var(--white);
        border-radius: var(--radius-md);
        padding: 20px 22px;
        border-left: 4px solid var(--green-mid);
        box-shadow: var(--shadow-sm);
        height: 100%;
        transition: box-shadow 0.2s, transform 0.2s;
    }

    .home-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .home-card h4 {
        margin: 0 0 8px 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary) !important;
    }

    .home-card p {
        margin: 0;
        color: #556b5e !important;
        font-size: 0.87rem;
        line-height: 1.65;
    }

    /* ── Disclaimer ── */
    .home-disclaimer {
        background: #fffbec;
        border-radius: var(--radius-sm);
        padding: 16px 20px;
        border-left: 4px solid #f59e0b;
        font-size: 0.85rem;
        color: #78350f !important;
        line-height: 1.65;
        margin-top: 28px;
    }

    .home-disclaimer strong {
        color: #78350f !important;
    }

    /* ── Input hint ── */
    .input-hint {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.8rem;
        margin: 0 0 6px 0;
        padding: 6px 12px;
        background: rgba(45,106,79,0.06);
        border-radius: 8px;
        border: 1px dashed rgba(45,106,79,0.2);
    }

    /* ── Chat header ── */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }

    .chat-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #52b788;
        box-shadow: 0 0 0 3px rgba(82,183,136,0.2);
    }

    /* ── Divider ── */
    .styled-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 20px 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE ---
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "router" not in st.session_state:
    st.session_state.router = None


# --- LOAD ROUTER ---
@st.cache_resource
def load_router():
    return AssistantRouter()


ROUTE_LABELS = {
    "rag":        ("🩺", "Base médicale"),
    "calculator": ("🧮", "Calculateur"),
    "weather":    ("🌤️", "Météo"),
    "web_search": ("🔍", "Recherche web"),
    "chat":       ("💬", "Conversation"),
}


# --- HOME PAGE ---
def render_home():
    # Hero
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a3c34 0%, #2d6a4f 100%);
        border-radius: 18px;
        padding: 48px 40px 44px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -30px; right: -30px;
            width: 180px; height: 180px; border-radius: 50%;
            background: rgba(255,255,255,0.04);
        "></div>
        <div style="
            position: absolute; bottom: -50px; left: 60px;
            width: 240px; height: 240px; border-radius: 50%;
            background: rgba(255,255,255,0.03);
        "></div>
        <div style="font-size: 2.4rem; margin-bottom: 6px;">🩺</div>
        <div style="
            font-family: 'DM Serif Display', serif;
            font-size: 2.5rem;
            color: white;
            margin-bottom: 14px;
            line-height: 1.2;
        ">Medical Assistant</div>
        <div style="
            font-size: 1rem;
            color: rgba(255,255,255,0.82);
            line-height: 1.75;
            max-width: 480px;
        ">
            Posez vos questions médicales en langage naturel — symptômes, conditions,
            médicaments — et obtenez des réponses basées sur des sources vérifiées.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("Démarrer une conversation", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    # Section title
    st.markdown("""
    <div style="
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: #1a3c34;
        margin: 32px 0 16px 0;
    ">Comment ça fonctionne</div>
    """, unsafe_allow_html=True)

    # Cards
    col1, col2 = st.columns(2)
    cards = [
        ("🗣️", "Posez une question",
         "Décrivez vos symptômes ou posez une question en français ou en anglais, librement."),
        ("💬", "Mémoire conversationnelle",
         "L'assistant se souvient des 5 derniers échanges pour des questions de suivi naturelles."),
        ("🔀", "Routage intelligent",
         "Détecte automatiquement s'il faut consulter la base médicale, calculer, météo ou simplement discuter."),
        ("🔒", "Local & privé",
         "Tout s'exécute sur votre machine. Aucune donnée n'est envoyée à des serveurs externes."),
    ]

    for i, (icon, title, desc) in enumerate(cards):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="home-card" style="margin-bottom: 14px;">
                <h4>{icon} &nbsp;{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="home-disclaimer">
        <strong>Avertissement</strong><br>
        Cet assistant est fourni à titre informatif uniquement et ne remplace pas un avis médical professionnel.
        Consultez toujours un professionnel de santé qualifié pour tout diagnostic ou traitement.
    </div>
    """, unsafe_allow_html=True)


# --- CHAT PAGE ---
def render_chat():
    # Header row
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("""
        <div class="chat-header">
            <div class="chat-dot"></div>
            <h2 style="margin:0; font-size:1.45rem;">Medical Assistant</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("⬅ Accueil"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    # Load router
    if st.session_state.router is None:
        with st.spinner("Chargement de l'assistant..."):
            try:
                st.session_state.router = load_router()
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")
                st.stop()

    # Empty state
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 40px 20px;
            color: #7a7a7a;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🩺</div>
            <div style="font-family: 'DM Serif Display', serif; font-size: 1.2rem;
                        color: #1a3c34; margin-bottom: 8px;">
                Comment puis-je vous aider ?
            </div>
            <div style="font-size: 0.88rem; line-height: 1.7; max-width: 360px; margin: 0 auto;">
                Décrivez vos symptômes, posez une question sur un médicament,
                ou demandez quand consulter un médecin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("route"):
                route = message["route"]
                icon, label = ROUTE_LABELS.get(route, ("💬", route))
                st.markdown(
                    f'<span class="route-tag route-{route}">{icon} {label}</span>',
                    unsafe_allow_html=True
                )

            # Safe text rendering with overflow fix
            content = message["content"] or ""
            st.markdown(
                f'<div style="word-wrap:break-word; overflow-wrap:break-word; '
                f'white-space:pre-wrap; color:#2e2e2e; line-height:1.7;">{content}</div>',
                unsafe_allow_html=True
            )

            if message["role"] == "assistant" and message.get("sources"):
                st.markdown('<div class="sources-block"><div class="sources-label">Sources</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="source-tag">📄 {s}</span>' for s in message["sources"]])
                st.markdown(tags + "</div>", unsafe_allow_html=True)

    # Clear button
    if st.session_state.chat_history:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Effacer la conversation"):
            st.session_state.chat_history = []
            if st.session_state.router:
                st.session_state.router.memory.clear()
            st.rerun()

    # Input hint
    st.markdown("""
    <p class="input-hint">
        🔍 &nbsp;Décrivez vos symptômes, posez une question sur un médicament, ou demandez simplement…
    </p>
    """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ex : J'ai de la fièvre depuis 3 jours, que faire ?")

    if user_input:
        with st.chat_message("user"):
            st.markdown(
                f'<div style="word-wrap:break-word; overflow-wrap:break-word; '
                f'white-space:pre-wrap; color:#2e2e2e; line-height:1.7;">{user_input}</div>',
                unsafe_allow_html=True
            )

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "route": None,
            "sources": [],
        })

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours…"):
                result = st.session_state.router.route(user_input)

            route   = result.get("type", "chat")
            answer  = result.get("answer", "")
            sources = result.get("sources", [])

            icon, label = ROUTE_LABELS.get(route, ("💬", route))
            st.markdown(
                f'<span class="route-tag route-{route}">{icon} {label}</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="word-wrap:break-word; overflow-wrap:break-word; '
                f'white-space:pre-wrap; color:#2e2e2e; line-height:1.7;">{answer}</div>',
                unsafe_allow_html=True
            )

            if sources:
                st.markdown('<div class="sources-block"><div class="sources-label">Sources</div>', unsafe_allow_html=True)
                tags = "".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
                st.markdown(tags + "</div>", unsafe_allow_html=True)

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
