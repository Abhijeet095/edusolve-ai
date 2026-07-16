import streamlit as st
import os
import base64
import html
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    CHAT_MODEL, VISION_MODEL,
    CHAT_TEMPERATURE, VISION_TEMPERATURE,
    EMBEDDING_MODEL, DB_PATH, RETRIEVER_K, MAX_RETRIEVAL_DISTANCE,
    GRADES, SUBJECTS,
    MEMORY_WINDOW_K,
    CONFIDENCE_OPTIONS,
    APP_TITLE, APP_ICON, APP_LAYOUT, WELCOME_TEXT
)
from prompts import (
    build_system_prompt,
    build_rewrite_prompt,
    generate_followups,
    generate_practice_question,
    is_question_on_subject,
    build_off_subject_message
)

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.set_page_config(page_title="EduSolve AI", page_icon="🎓")
    st.error(
        "⚠️ **GROQ_API_KEY not found.**\n\n"
        "Please create a `.env` file in the project folder with your Groq API key:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n\n"
        "Get your free API key at [console.groq.com](https://console.groq.com)"
    )
    st.stop()

# =============================================================================
# PAGE CONFIGURATION
# — Changed layout to "centered" for mobile compatibility
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon=APP_ICON,
    initial_sidebar_state="collapsed"
)

# =============================================================================
# MOBILE-OPTIMISED CSS
# Key changes vs original:
#   • font-size: 16px on inputs  → prevents iOS auto-zoom
#   • min-height: 44px on buttons → Apple HIG minimum tap target
#   • width: 100% on buttons      → full-width tap areas on small screens
#   • flex-direction: column on radio → stacks grade options vertically
#   • max-width: 100% on images   → no overflow on narrow phones
#   • min-width: 280px on sidebar → readable when opened on mobile
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { --blue:#2563EB; --green:#16A34A; --orange:#F59E0B; --ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --surface:#FFFFFF; --page:#F8FAFC; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size:16px !important; }
[data-testid="stAppViewContainer"] { background:var(--page); color:var(--ink); }
.stApp, [data-testid="stAppViewContainer"] * { color:var(--ink); }
header[data-testid="stHeader"] { background:var(--page) !important; border-bottom:1px solid var(--line); }
header[data-testid="stHeader"] * { color:var(--muted) !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width:none !important; width:100% !important; padding:1.5rem 3rem 6.5rem; }

/* Brand, headings, and reusable cards */
.brand-lockup { text-align:center; margin:1.2rem 0 1rem; }
.brand-mark { width:72px; height:72px; display:inline-flex; align-items:center; justify-content:center; border-radius:21px; background:#DBEAFE; color:var(--blue); font-size:2.1rem; margin-bottom:.75rem; }
.brand-title { font-size:2rem; font-weight:800; letter-spacing:-.05em; color:var(--ink); margin:0; }
.brand-subtitle { color:var(--blue); font-size:1rem; font-weight:700; margin:.35rem 0 .7rem; }
.brand-copy { color:var(--muted); font-size:.95rem; line-height:1.65; margin:0 auto; max-width:455px; }
.page-kicker, .page-title, .page-copy { max-width:920px; margin-left:auto; margin-right:auto; }
.page-kicker { color:var(--blue); font-weight:700; font-size:.76rem; text-transform:uppercase; letter-spacing:.1em; margin-top:0; margin-bottom:.3rem; }
.page-title { color:var(--ink); font-size:1.45rem; font-weight:800; letter-spacing:-.04em; margin-top:0; margin-bottom:0; }
.page-copy { color:var(--muted); margin-top:.35rem; margin-bottom:.9rem; }
.section-label { color:var(--ink); font-size:.94rem; font-weight:700; margin:.1rem 0 .7rem; }
.status-pill { display:inline-block; background:#DCFCE7; color:#166534; border-radius:999px; padding:.4rem .7rem; font-size:.75rem; font-weight:700; }
.sidebar-heading { color:var(--muted); font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.1em; margin:1.1rem 0 .55rem; }
.sidebar-name { color:var(--ink); font-size:1.05rem; font-weight:800; margin:0 0 .15rem; }
.sidebar-meta { color:var(--muted); font-size:.84rem; margin:0; }
.upload-title { color:var(--ink); font-size:.92rem; font-weight:700; margin:0 0 .2rem; }
.upload-copy { color:var(--muted); font-size:.8rem; margin:0 0 .55rem; }
.empty-state { text-align:center; max-width:640px; margin:1.5rem auto .8rem; padding:1.15rem 1.25rem .8rem; }
.empty-icon { display:inline-flex; align-items:center; justify-content:center; width:60px; height:60px; border-radius:18px; background:#DBEAFE; font-size:1.7rem; }
.empty-state h2 { color:var(--ink); font-size:1.55rem; letter-spacing:-.04em; margin:.65rem 0 .25rem; }
.empty-state p { color:var(--muted); margin:0; }
.hint-banner { border:1px solid #BFDBFE; background:#EFF6FF; color:#1D4ED8; border-radius:14px; padding:.8rem 1rem; margin:.2rem 0 1rem; }
.fallback-warning { background:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid var(--orange); padding:.75rem 1rem; border-radius:12px; font-size:.86rem; margin-top:.75rem; color:#92400E; }

/* Streamlit controls */
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface); border:1px solid var(--line); border-radius:18px; box-shadow:0 8px 24px rgba(15,23,42,.04); color:var(--ink) !important; }
[data-testid="stVerticalBlockBorderWrapper"] *, [data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .sidebar-meta, .upload-copy { color:var(--muted) !important; }
[data-testid="stButton"] button, .stButton > button { min-height:46px; width:100%; border-radius:12px; border:1px solid #CBD5E1; background:#FFF; color:#1E293B !important; font-weight:600; font-size:.9rem; transition:all .18s ease; white-space:normal; }
[data-testid="stButton"] button:hover, .stButton > button:hover { border-color:#93C5FD; color:#1D4ED8 !important; background:#EFF6FF; transform:translateY(-1px); }
[data-testid="stButton"] button[kind="primary"], .stButton > button[kind="primary"] { background:var(--blue); border-color:var(--blue); color:#FFF !important; box-shadow:0 6px 14px rgba(37,99,235,.2); }
[data-testid="stButton"] button[kind="primary"] *, .stButton > button[kind="primary"] * { color:#FFF !important; }
[data-testid="stButton"] button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover { background:#1D4ED8; color:#FFF !important; }
[data-testid="stTextInput"] input, [data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea::placeholder { color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; opacity:1; }
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stFileUploader"] section { border-color:#CBD5E1 !important; border-radius:12px !important; background:#FFF !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] *, [data-testid="stFileUploader"] * { color:var(--ink) !important; }
[data-testid="stTextInput"] input { min-height:46px; }
[data-testid="stRadio"] > div { gap:.6rem; }
[data-testid="stRadio"] label { padding:.45rem .1rem; }

/* Sidebar */
[data-testid="stSidebar"] { background:#FFF; border-right:1px solid var(--line); min-width:285px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow:none; border-radius:14px; }
[data-testid="stSidebar"] .stButton > button { min-height:42px; font-size:.84rem; }

/* The main area uses the full viewport after Streamlit collapses the sidebar. */
[data-testid="stMain"] { min-width:0; }
[data-testid="stMain"] [data-testid="stMainBlockContainer"] { max-width:none !important; margin:0 !important; }

/* Chat */
[data-testid="stChatMessage"] { max-width:920px; border:0; padding:0; margin:0 auto .8rem; background:transparent; }
[data-testid="stChatMessage"] > div { border-radius:18px; padding:1rem 1.1rem; border:1px solid var(--line); box-shadow:0 4px 14px rgba(15,23,42,.035); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div { background:#F8FAFC; border-color:#E2E8F0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div { background:#EFF6FF; border-color:#BFDBFE; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height:1.7; }
[data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol { padding-left:1.35rem; }
[data-testid="stChatMessage"] strong { color:#0F172A; }
[data-testid="stBottomBlockContainer"] { background:var(--page) !important; border-top:1px solid var(--line); }
[data-testid="stChatInput"] { --background-color:#FFF !important; --secondary-background-color:#FFF !important; max-width:920px; margin:0 auto; border:1px solid #BFDBFE !important; border-radius:18px; box-shadow:0 8px 22px rgba(37,99,235,.09); background:#FFF !important; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb="base-input"], [data-testid="stChatInput"] [data-baseweb="textarea"], [data-testid="stChatInput"] textarea { background:#FFF !important; }
[data-testid="stChatInput"] textarea { color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
textarea[data-testid="stChatInputTextArea"] { min-height:62px !important; padding:.8rem .9rem !important; font-size:16px !important; }

img { max-width:100% !important; height:auto !important; border-radius:14px; }
@media (max-width: 760px) { .block-container, [data-testid="stMainBlockContainer"] { padding:1rem 1rem 6rem; } .brand-title { font-size:1.7rem; } .page-title { font-size:1.3rem; } [data-testid="stSidebar"] { min-width:0 !important; } [data-testid="stChatMessage"] { margin-bottom:.7rem; } }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE AI MODELS (cached — load only once)
# =============================================================================

@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vision_ai = ChatGroq(
        model_name=VISION_MODEL,
        groq_api_key=groq_key,
        temperature=VISION_TEMPERATURE
    )
    chat_ai = ChatGroq(
        model_name=CHAT_MODEL,
        groq_api_key=groq_key,
        temperature=CHAT_TEMPERATURE
    )
    return embeddings, vision_ai, chat_ai

embeddings_model, vision_ai, chat_ai = init_models()

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# =============================================================================
# PARALLEL POST-ANSWER GENERATOR
# =============================================================================

def generate_post_answer_content(
    answer: str,
    subject: str,
    grade: str,
) -> tuple[list[str], str]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_followups = executor.submit(
            generate_followups, answer, subject, grade, chat_ai
        )
        future_practice = executor.submit(
            generate_practice_question, answer, subject, grade, chat_ai
        )
        followups = future_followups.result()
        practice = future_practice.result()
    return followups, practice

# =============================================================================
# STREAMING HELPER
# =============================================================================

def get_streaming_answer(
    vector_store: Chroma,
    memory: list,
    prompt: str,
    grade: str,
    subject: str,
    hint_mode: bool
) -> tuple[str, list, bool]:
    chat_history = memory[-(MEMORY_WINDOW_K * 2):]

    if chat_history:
        rewritten = chat_ai.invoke(
            build_rewrite_prompt(str(chat_history), prompt)
        ).content.strip()
    else:
        rewritten = prompt

    # Chroma's raw distance API is used deliberately. Its built-in relevance
    # mapping depends on the collection metric and can produce invalid scores.
    # Lower distances are more similar.
    scored_docs = vector_store.similarity_search_with_score(
        rewritten,
        k=RETRIEVER_K,
        filter={
            "$and": [
                {"grade": grade},
                {"subject": subject}
            ]
        }
    )
    docs = [
        document for document, distance in scored_docs
        if distance <= MAX_RETRIEVAL_DISTANCE
    ]
    context = "\n\n".join([d.page_content for d in docs])
    used_fallback = not bool(context.strip())

    student_name = st.session_state.get("student_name", "")
    system_text = build_system_prompt(grade, subject, hint_mode, student_name).replace(
        "{context}", context if context else "[No specific textbook content found. Use general knowledge.]"
    )

    history_text = f"Chat history:\n{chat_history}\n\n" if chat_history else ""
    final_messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"{history_text}Student's question: {prompt}")
    ]

    full_answer = st.write_stream(chat_ai.stream(final_messages))

    memory.extend([HumanMessage(content=prompt), AIMessage(content=full_answer)])
    del memory[:-(MEMORY_WINDOW_K * 2)]

    return full_answer, docs, used_fallback

# =============================================================================
# MEMORY HELPERS — Subject-scoped
# =============================================================================

def memory_key(subject: str) -> str:
    return f"memory_{subject}"

def messages_key(subject: str) -> str:
    return f"messages_{subject}"

def get_memory(subject: str) -> list:
    """Return the session-local message history for one subject.

    A plain list of LangChain messages replaces the deprecated memory classes.
    """
    key = memory_key(subject)
    if key not in st.session_state or not isinstance(st.session_state[key], list):
        st.session_state[key] = []
    return st.session_state[key]

def get_messages(subject: str) -> list:
    key = messages_key(subject)
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]

def clear_subject(subject: str):
    mem_key = memory_key(subject)
    msg_key = messages_key(subject)
    if mem_key in st.session_state:
        del st.session_state[mem_key]
    if msg_key in st.session_state:
        st.session_state[msg_key] = []

def clear_all_subjects():
    for subj in SUBJECTS:
        clear_subject(subj)

# =============================================================================
# BUILD VECTOR STORE
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_vector_store() -> Chroma:
    """Return the shared, read-only textbook vector store.

    Conversation memory must never be cached here: Streamlit resource caches are
    shared across sessions, while student memory belongs only in session state.
    """
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings_model)

# =============================================================================
# SESSION STATE
# =============================================================================

if "student_name"    not in st.session_state: st.session_state.student_name    = ""
if "grade_locked"    not in st.session_state: st.session_state.grade_locked    = False
if "last_uploaded"   not in st.session_state: st.session_state.last_uploaded   = None
if "staged_image"    not in st.session_state: st.session_state.staged_image    = None
if "followups"       not in st.session_state: st.session_state.followups       = []
if "pending_prompt"  not in st.session_state: st.session_state.pending_prompt  = None
if "practice_q"      not in st.session_state: st.session_state.practice_q      = ""
if "show_confidence" not in st.session_state: st.session_state.show_confidence = False
if "rating_log"      not in st.session_state: st.session_state.rating_log      = []
if "active_subject"  not in st.session_state: st.session_state.active_subject  = None
if "ui_sidebar_open" not in st.session_state: st.session_state.ui_sidebar_open = True
if "ui_subject"      not in st.session_state: st.session_state.ui_subject      = SUBJECTS[0]
if "ui_hint_mode"    not in st.session_state: st.session_state.ui_hint_mode    = False

# =============================================================================
# STEP 1: WELCOME / GRADE SELECTION
# — Removed horizontal=True from radio so grades stack vertically on mobile
# =============================================================================

if not st.session_state.grade_locked:
    _, landing_column, _ = st.columns([1, 1.35, 1])
    with landing_column:
        st.markdown("""
        <div class="brand-lockup">
            <div class="brand-mark">✦</div>
            <h1 class="brand-title">EduSolve AI</h1>
            <p class="brand-subtitle">Your Personal AI Tutor</p>
            <p class="brand-copy">Learn Mathematics, Science and English with an AI tutor that understands your textbooks and guides you step by step.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<p class='section-label'>Start your learning session</p>", unsafe_allow_html=True)
            name_input = st.text_input(
                "Student name",
                placeholder="e.g. Rohan",
                max_chars=40
            )
            grade = st.radio("Select your standard", GRADES)

            if st.button("Start Learning", type="primary", use_container_width=True):
                if not name_input.strip():
                    st.warning("Please enter your name to continue!")
                else:
                    st.session_state.student_name = name_input.strip().capitalize()
                    st.session_state.grade = grade
                    st.session_state.grade_locked = True
                    st.rerun()

# =============================================================================
# STEP 2: MAIN CLASSROOM INTERFACE
# =============================================================================

else:
    # Page-level navigation replaces st.sidebar, so a collapsed navigation rail
    # never leaves a hidden Streamlit sidebar width behind.
    if st.session_state.ui_sidebar_open:
        sidebar_column, main_column = st.columns([1.25, 4.75], gap="medium")
    else:
        sidebar_column, main_column = st.columns([0.28, 5.72], gap="small")

    with sidebar_column:
        menu_label = "‹" if st.session_state.ui_sidebar_open else "☰"
        if st.button(menu_label, key="ui_sidebar_toggle", help="Collapse or expand navigation"):
            st.session_state.ui_sidebar_open = not st.session_state.ui_sidebar_open
            st.rerun()

        if st.session_state.ui_sidebar_open:
            st.markdown("""
            <div style="padding:.35rem 0 .1rem">
                <div class="brand-mark" style="width:50px;height:50px;border-radius:15px;font-size:1.4rem;margin:0 0 .5rem">✦</div>
                <p class="sidebar-name">EduSolve AI</p>
                <p class="sidebar-meta">Personal learning space</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<p class='sidebar-heading'>Student</p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"<p class='sidebar-name'>{html.escape(st.session_state.student_name)}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='sidebar-meta'>{st.session_state.grade}</p>", unsafe_allow_html=True)

            st.markdown("<p class='sidebar-heading'>Current subject</p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.selectbox("Subject", SUBJECTS, key="ui_subject", label_visibility="collapsed")

            st.markdown("<p class='sidebar-heading'>Learning options</p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.checkbox(
                    "Hint mode",
                    key="ui_hint_mode",
                    help="Agent gives hints instead of full answers."
                )
                st.caption("Guided clues instead of direct answers")

            st.markdown("<p class='sidebar-heading'>Question image</p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<p class='upload-title'>Upload question image</p><p class='upload-copy'>Drag and drop a clear photo, or browse files.</p>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Upload question image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
                if uploaded_file and st.session_state.last_uploaded != uploaded_file.name:
                    st.session_state.last_uploaded = uploaded_file.name
                    st.session_state.staged_image = Image.open(uploaded_file)
                    st.rerun()

            st.markdown("<p class='sidebar-heading'>Session</p>", unsafe_allow_html=True)
            with st.container(border=True):
                exchange_count = len(get_memory(st.session_state.ui_subject)) // 2
                st.caption(f"Memory: {exchange_count}/{MEMORY_WINDOW_K} exchanges")
                if st.button("Clear current chat", use_container_width=True):
                    clear_subject(st.session_state.ui_subject)
                    st.session_state.staged_image = None
                    st.session_state.followups = []
                    st.session_state.practice_q = ""
                    st.session_state.show_confidence = False
                    st.rerun()

                if st.button("Change grade / exit session", use_container_width=True):
                    st.session_state.grade_locked = False
                    st.session_state.student_name = ""
                    st.session_state.staged_image = None
                    st.session_state.followups = []
                    st.session_state.practice_q = ""
                    st.session_state.show_confidence = False
                    st.session_state.rating_log = []
                    clear_all_subjects()
                    st.rerun()

    subject = st.session_state.ui_subject
    hint_mode = st.session_state.ui_hint_mode
    if st.session_state.active_subject != subject:
        st.session_state.active_subject = subject
        st.session_state.show_confidence = False
        st.session_state.followups = []
        st.session_state.practice_q = ""

    # --- Main Chat Area ---
    display_grade = st.session_state.grade.replace("_", " ").upper()
    display_subject = {"Math": "Mathematics"}.get(subject, subject)
    main_column.markdown(f"""
    <p class="page-kicker">{display_grade} • {display_subject.upper()}</p>
    <h1 class="page-title">Learn with confidence, {html.escape(st.session_state.student_name)}</h1>
    <p class="page-copy">Ask a question, upload an image, or choose a prompt to begin.</p>
    """, unsafe_allow_html=True)

    if hint_mode:
        main_column.markdown("<div class='hint-banner'><strong>Hint mode is on.</strong> You’ll get guided clues before the full answer.</div>", unsafe_allow_html=True)

    vector_store = get_vector_store()
    memory = get_memory(subject)

    subject_messages = get_messages(subject)

    # --- Display Chat History ---
    for msg in subject_messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with main_column.chat_message(msg["role"], avatar=avatar):
            if "image" in msg:
                # ← CHANGED: use_column_width=True instead of width=350
                # Prevents images from overflowing narrow phone screens
                st.image(msg["image"], use_column_width=True)
            # Chat content can come from the model, so never render it as HTML.
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"📚 Source: {', '.join(msg['sources'])}")

    if not subject_messages and not st.session_state.staged_image:
        main_column.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <h2>Suggested prompts</h2>
            <p>Choose a starting point or ask your own question below.</p>
        </div>
        """, unsafe_allow_html=True)
        _, prompt_area, _ = main_column.columns([1, 2.2, 1])
        with prompt_area:
            prompt_columns = st.columns(2)
            starter_prompts = [
                "Explain Algebra",
                "Summarize today's lesson",
                "Give me practice questions",
                "Explain with examples",
            ]
            for index, starter_prompt in enumerate(starter_prompts):
                with prompt_columns[index % 2]:
                    if st.button(starter_prompt, key=f"starter_{index}_{subject}", use_container_width=True):
                        st.session_state.pending_prompt = starter_prompt
                        st.rerun()

    # -------------------------------------------------------------------------
    # CONFIDENCE CHECK
    # — CHANGED: replaced st.columns([2, 2, 3, 5]) with vertical button stack
    #   Columns collapse to tiny unusable buttons on phones.
    #   Vertical stack gives each button a full-width 44px tap target.
    # -------------------------------------------------------------------------

    if st.session_state.show_confidence:
        with main_column.container(border=True):
            st.markdown("<p class='section-label'>How confident do you feel?</p>", unsafe_allow_html=True)
            for i, (label, rating, should_reexplain) in enumerate(CONFIDENCE_OPTIONS):
                if st.button(label, key=f"conf_{i}_{subject}", use_container_width=True):
                    st.session_state.rating_log.append({
                        "student": st.session_state.student_name,
                        "subject": subject,
                        "grade": st.session_state.grade,
                        "rating": rating,
                        "label": label
                    })
                    st.session_state.show_confidence = False
                    if should_reexplain:
                        st.session_state.pending_prompt = (
                            "I didn't understand that well. "
                            "Can you explain it again in a simpler way with a different example?"
                        )
                    st.rerun()

    # -------------------------------------------------------------------------
    # PRACTICE QUESTION + FOLLOW-UPS
    # — CHANGED: replaced dynamic st.columns() grid with vertical button stack
    #   Follow-up text can be long — it wraps badly in narrow columns on mobile.
    #   Vertical stack keeps each suggestion fully readable.
    # -------------------------------------------------------------------------

    show_pq = bool(st.session_state.practice_q)
    show_fu = bool(st.session_state.followups)

    if show_pq or show_fu:
        with main_column.container(border=True):
            st.markdown("<p class='section-label'>Continue learning</p>", unsafe_allow_html=True)

            if show_pq:
                if st.button(
                    "Practice question",
                    key=f"pq_btn_{subject}",
                    help=st.session_state.practice_q,
                    type="primary",
                    use_container_width=True
                ):
                    pq = st.session_state.practice_q
                    st.session_state.practice_q = ""
                    st.session_state.followups = []
                    st.session_state.show_confidence = False
                    st.session_state.pending_prompt = (
                        f"Give me this practice question and wait for my answer before explaining: {pq}"
                    )
                    st.rerun()

            if show_fu:
                for i, suggestion in enumerate(st.session_state.followups):
                    if st.button(
                        suggestion,
                        key=f"followup_{i}_{subject}",
                        use_container_width=True
                    ):
                        st.session_state.pending_prompt = suggestion
                        st.session_state.followups = []
                        st.rerun()

    # --- Staged Image Preview ---
    if st.session_state.staged_image and not any(
        m.get("image_id") == st.session_state.last_uploaded
        for m in subject_messages
    ):
        with main_column.container(border=True):
            st.success("Image ready. Ask your question below.")
            st.image(st.session_state.staged_image, use_column_width=True)

    # --- Chat Input ---
    typed_prompt = main_column.chat_input(f"Ask a {subject} question…")
    prompt = typed_prompt or st.session_state.pop("pending_prompt", None)

    if prompt:
        st.session_state.followups = []
        user_msg = {"role": "user", "content": prompt}

        # =====================================================================
        # PATH A: Image + Text → Vision Model
        # =====================================================================
        if st.session_state.staged_image:
            chat_history_for_check = str(get_memory(subject))
            if not is_question_on_subject(prompt, subject, chat_history_for_check, chat_ai):
                subject_messages.append(user_msg)
                with main_column.chat_message("user", avatar="🧑‍🎓"):
                    # ← CHANGED: use_column_width=True
                    st.image(st.session_state.staged_image, use_column_width=True)
                    st.markdown(prompt)
                refusal = build_off_subject_message(subject, prompt, chat_ai)
                with main_column.chat_message("assistant", avatar="🤖"):
                    st.markdown(refusal)
                subject_messages.append({"role": "assistant", "content": refusal})
                st.session_state.staged_image = None
                st.rerun()

            user_msg["image"] = st.session_state.staged_image
            user_msg["image_id"] = st.session_state.last_uploaded
            subject_messages.append(user_msg)

            with main_column.chat_message("user", avatar="🧑‍🎓"):
                # ← CHANGED: use_column_width=True
                st.image(st.session_state.staged_image, use_column_width=True)
                st.markdown(prompt)

            with main_column.chat_message("assistant", avatar="🤖"):
                try:
                    b64 = encode_image(st.session_state.staged_image)
                    vision_system = build_system_prompt(
                        st.session_state.grade, subject, hint_mode,
                        st.session_state.student_name
                    ).replace(
                        "{context}",
                        "[No textbook context for image questions. Use general knowledge.]"
                    )

                    vision_messages = [SystemMessage(content=vision_system)]
                    for mem_msg in get_memory(subject):
                        if isinstance(mem_msg, HumanMessage):
                            vision_messages.append(HumanMessage(content=mem_msg.content))
                        elif isinstance(mem_msg, AIMessage):
                            vision_messages.append(AIMessage(content=mem_msg.content))

                    vision_messages.append(
                        HumanMessage(content=[
                            {"type": "text", "text": f"Student's question: {prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        ])
                    )

                    answer = st.write_stream(vision_ai.stream(vision_messages))

                    memory = get_memory(subject)
                    memory.extend([HumanMessage(content=prompt), AIMessage(content=answer)])
                    del memory[:-(MEMORY_WINDOW_K * 2)]
                    subject_messages.append({"role": "assistant", "content": answer})
                    st.session_state.staged_image = None

                    st.markdown(
                        "<div class='fallback-warning'>⚠️ This answer is based on general knowledge, "
                        "not your specific textbook. Please verify with your teacher or notes.</div>",
                        unsafe_allow_html=True
                    )

                    with st.spinner("Preparing your next learning steps…"):
                        fu, pq = generate_post_answer_content(
                            answer, subject, st.session_state.grade
                        )
                    st.session_state.followups = fu
                    st.session_state.practice_q = pq
                    st.session_state.show_confidence = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

        # =====================================================================
        # PATH B: Text-only → Streaming RAG answer
        # =====================================================================
        else:
            chat_history_for_check = str(get_memory(subject))
            if not is_question_on_subject(prompt, subject, chat_history_for_check, chat_ai):
                subject_messages.append(user_msg)
                with main_column.chat_message("user", avatar="🧑‍🎓"):
                    st.markdown(prompt)
                refusal = build_off_subject_message(subject, prompt, chat_ai)
                with main_column.chat_message("assistant", avatar="🤖"):
                    st.markdown(refusal)
                subject_messages.append({"role": "assistant", "content": refusal})
                st.rerun()

            subject_messages.append(user_msg)
            with main_column.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(prompt)

            with main_column.chat_message("assistant", avatar="🤖"):
                try:
                    answer, source_docs, used_fallback = get_streaming_answer(
                        vector_store, memory, prompt,
                        st.session_state.grade, subject, hint_mode
                    )

                    if source_docs:
                        sources = sorted({
                            os.path.basename(d.metadata.get('source', 'Unknown'))
                            for d in source_docs
                        })
                        st.caption(f"📚 Source: {', '.join(sources)}")

                    if used_fallback:
                        st.markdown(
                            "<div class='fallback-warning'>⚠️ I couldn't find this topic in your "
                            "textbook. This answer is from general knowledge — please verify "
                            "with your teacher or notes.</div>",
                            unsafe_allow_html=True
                        )

                    subject_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources if source_docs else []
                    })

                    with st.spinner("Preparing your next learning steps…"):
                        fu, pq = generate_post_answer_content(
                            answer, subject, st.session_state.grade
                        )
                    st.session_state.followups = fu
                    st.session_state.practice_q = pq
                    st.session_state.show_confidence = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error fetching answer: {e}")
