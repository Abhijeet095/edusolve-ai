import streamlit as st
import os
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain       # ← replaces RetrievalQA
from langchain.memory import ConversationBufferWindowMemory     # ← new: chat memory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    CHAT_MODEL, VISION_MODEL,
    CHAT_TEMPERATURE, VISION_TEMPERATURE,
    EMBEDDING_MODEL, DB_PATH,
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

# Validate API key at startup — show a friendly error before anything else loads.
# Without this, the app loads fine but crashes on the student's first question
# with a raw technical error they won't understand.
if not groq_key:
    st.set_page_config(page_title="EduSolve AI", page_icon="🎓")
    st.error(
        "⚠️ **GROQ_API_KEY not found.**\n\n"
        "Please create a `.env` file in the project folder with your Groq API key:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n\n"
        "Get your free API key at [console.groq.com](https://console.groq.com)"
    )
    st.stop()


# build_system_prompt(), generate_followups(), generate_practice_question()
# moved to prompts.py — imported at top of file.


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout=APP_LAYOUT,
    page_icon=APP_ICON
)

st.markdown("""
    <style>
    .stChatMessage { border-radius: 8px; border: 1px solid #e0e0e0; padding: 15px; margin-bottom: 10px; }
    .source-tag { color: #28a745; font-weight: bold; font-size: 0.8rem; margin-top: 8px; display: block; }
    .fallback-warning { background: #fff8e1; border-left: 4px solid #f9a825; padding: 8px 12px;
                        border-radius: 4px; font-size: 0.85rem; margin-top: 8px; }
    p { line-height: 1.6; }
    div[data-testid="stHorizontalBlock"] button {
        font-size: 0.78rem !important;
        padding: 4px 8px !important;
        min-height: 0 !important;
        height: auto !important;
        white-space: normal !important;
        line-height: 1.3 !important;
    }
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
# After every answer, we need two LLM calls:
#   1. generate_followups()         — 3 follow-up question suggestions
#   2. generate_practice_question() — 1 exam-style practice question
#
# Previously these ran sequentially, adding ~4-6s of blank wait after streaming.
# Now both run simultaneously using ThreadPoolExecutor — total wait time drops
# to roughly the slower of the two calls instead of the sum of both.
#
# A spinner is shown during this wait so the student sees feedback.
# =============================================================================
def generate_post_answer_content(
    answer: str,
    subject: str,
    grade: str,
) -> tuple[list[str], str]:
    """
    Runs generate_followups() and generate_practice_question() in parallel.
    Returns (followups_list, practice_question_string).
    Falls back gracefully — if either call fails, returns empty value for that one.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_followups = executor.submit(
            generate_followups, answer, subject, grade, chat_ai
        )
        future_practice = executor.submit(
            generate_practice_question, answer, subject, grade, chat_ai
        )
        followups = future_followups.result()
        practice  = future_practice.result()

    return followups, practice


# =============================================================================
# STREAMING HELPER
# =============================================================================
def get_streaming_answer(
    qa_chain,
    prompt: str,
    grade: str,
    subject: str,
    hint_mode: bool
) -> tuple[str, list, bool]:
    """
    Returns (full_answer_string, source_docs, used_fallback).
      - source_docs   -> reused for source tags, no second retrieval needed
      - used_fallback -> True when no textbook content was found
    """
    retriever    = qa_chain.retriever
    chat_history = qa_chain.memory.load_memory_variables({})["chat_history"]

    if chat_history:
        rewritten = chat_ai.invoke(
            build_rewrite_prompt(str(chat_history), prompt)
        ).content.strip()
    else:
        rewritten = prompt

    docs          = retriever.invoke(rewritten)
    context       = "\n\n".join([d.page_content for d in docs])
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
    qa_chain.memory.chat_memory.add_user_message(prompt)
    qa_chain.memory.chat_memory.add_ai_message(full_answer)

    return full_answer, docs, used_fallback


# =============================================================================
# MEMORY HELPERS — Subject-scoped
#
# Each subject gets its own memory and message list stored under a unique key:
#   memory_Math, memory_Science, memory_English
#   messages_Math, messages_Science, messages_English
#
# Switching subjects in the sidebar automatically loads that subject's history.
# "Clear Chat" only wipes the currently active subject.
# "Change Grade / Exit" wipes everything across all subjects.
# =============================================================================

# SUBJECTS and GRADES imported from config.py

def memory_key(subject: str) -> str:
    """Returns the session_state key for a subject's LangChain memory."""
    return f"memory_{subject}"

def messages_key(subject: str) -> str:
    """Returns the session_state key for a subject's chat display messages."""
    return f"messages_{subject}"

def get_memory(subject: str) -> ConversationBufferWindowMemory:
    """
    Returns the memory object for the given subject.
    Creates a fresh one if it doesn't exist yet.
    k=5 → last 5 exchange pairs remembered per subject.
    """
    key = memory_key(subject)
    if key not in st.session_state:
        st.session_state[key] = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_K,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return st.session_state[key]

def get_messages(subject: str) -> list:
    """Returns the display message list for the given subject."""
    key = messages_key(subject)
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]

def clear_subject(subject: str):
    """Wipes memory + messages for one subject. Called on Clear Chat."""
    mem_key = memory_key(subject)
    msg_key = messages_key(subject)
    if mem_key in st.session_state:
        del st.session_state[mem_key]
    if msg_key in st.session_state:
        st.session_state[msg_key] = []

def clear_all_subjects():
    """Wipes memory + messages for ALL subjects. Called on Change Grade / Exit."""
    for subj in SUBJECTS:
        clear_subject(subj)


# =============================================================================
# BUILD CHAIN — ConversationalRetrievalChain (memory-aware)
# Cached by grade + subject + hint_mode so it is NOT rebuilt on every
# Streamlit rerun. Cache is busted automatically when any of these change.
# NOTE: memory is NOT part of the cache key — get_memory() pulls it fresh
# from session_state each time, so memory updates still work correctly.
# =============================================================================
@st.cache_resource(show_spinner=False)
def build_qa_chain(grade: str, subject: str, hint_mode: bool):
    """
    How this chain works:
    1. Student asks "explain that again" (a vague follow-up)
    2. Chain internally rewrites it to a standalone question using chat history
       e.g. → "Explain the photosynthesis process again in simple terms"
    3. Retrieves relevant docs from Chroma using the rewritten question
    4. Answers using: system prompt + RAG context + chat history + question
    """
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings_model)
    retriever = db.as_retriever(
        search_kwargs={
            "filter": {
                "$and": [
                    {"grade": grade},
                    {"subject": subject}
                ]
            }
        }
    )

    # student_name not cached in chain — injected via system prompt at chain build time
    student_name = st.session_state.get("student_name", "")
    system_prompt_text = build_system_prompt(grade, subject, hint_mode, student_name)

    # Prompt for the final answer step
    # Receives {context} from RAG + {chat_history} from memory + {question}
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_text),
        HumanMessagePromptTemplate.from_template(
            "Chat history:\n{chat_history}\n\nStudent's question: {question}"
        )
    ])

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_ai,
        retriever=retriever,
        memory=None,                        # ← memory injected at call time, not cached
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key="answer",
        verbose=False
    )
    return chain


# =============================================================================
# SESSION STATE
# =============================================================================
if "student_name" not in st.session_state:   st.session_state.student_name = ""       # student's name entered on welcome screen
if "grade_locked" not in st.session_state:   st.session_state.grade_locked = False
if "last_uploaded" not in st.session_state:  st.session_state.last_uploaded = None
if "staged_image" not in st.session_state:   st.session_state.staged_image = None
if "followups" not in st.session_state:      st.session_state.followups = []      # current follow-up suggestions
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None  # prompt triggered by button click
if "practice_q" not in st.session_state:    st.session_state.practice_q = ""           # current practice question
if "show_confidence" not in st.session_state: st.session_state.show_confidence = False  # whether to show star rating
if "rating_log" not in st.session_state:    st.session_state.rating_log = []           # list of {subject, rating, topic}
if "active_subject" not in st.session_state: st.session_state.active_subject = None    # tracks current subject to detect switches


# =============================================================================
# STEP 1: WELCOME / GRADE SELECTION
# =============================================================================
if not st.session_state.grade_locked:
    st.title("🚀 EduSolve: Your Personal Tutor")
    st.markdown(WELCOME_TEXT)

    name_input = st.text_input(
        "Your name:",
        placeholder="e.g. Rohan",
        max_chars=40
    )

    grade = st.radio(
        "Select Your Standard:",
        GRADES,
        horizontal=True
    )

    if st.button("Start Session →"):
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
    # --- Sidebar ---
    with st.sidebar:
        st.title(f"📍 {st.session_state.student_name} — {st.session_state.grade}")
        subject = st.selectbox("Subject:", SUBJECTS)

        # Detect subject switch — clear UI state that belongs to the previous subject
        # so rating buttons and follow-up suggestions don't bleed across subjects
        if st.session_state.active_subject != subject:
            st.session_state.active_subject  = subject
            st.session_state.show_confidence = False
            st.session_state.followups       = []
            st.session_state.practice_q      = ""
        st.divider()

        hint_mode = st.checkbox(
            "💡 Hint Mode",
            help="Agent gives hints instead of full answers."
        )
        st.divider()

        uploaded_file = st.file_uploader("📷 Upload Question Photo", type=["jpg", "jpeg", "png"])

        if uploaded_file and st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.staged_image = Image.open(uploaded_file)
            st.rerun()

        st.divider()

        # Live memory counter — shows exchanges for the CURRENT subject only
        exchange_count = len(get_memory(subject).chat_memory.messages) // 2
        st.caption(f"🧠 {subject} Memory: {exchange_count}/5 exchanges stored")

        if st.button("🔄 Clear Chat"):
            clear_subject(subject)
            st.session_state.staged_image = None
            st.session_state.followups = []
            st.session_state.practice_q = ""
            st.session_state.show_confidence = False
            st.rerun()

        if st.button("🚪 Change Grade / Exit"):
            st.session_state.grade_locked = False
            st.session_state.student_name = ""
            st.session_state.staged_image = None
            st.session_state.followups = []
            st.session_state.practice_q = ""
            st.session_state.show_confidence = False
            st.session_state.rating_log = []
            clear_all_subjects()
            st.rerun()

    # --- Main Chat Area ---
    st.title(f"🤖 Hi {st.session_state.student_name}! {subject} Tutor — {st.session_state.grade}")

    if hint_mode:
        st.info("💡 **Hint Mode ON** — I'll guide you with clues instead of full answers.")

    qa_chain = build_qa_chain(st.session_state.grade, subject, hint_mode)
    # Memory is NOT cached — inject it fresh from session_state every rerun
    # so each subject keeps its own independent conversation history
    qa_chain.memory = get_memory(subject)

    # Get this subject's message list
    subject_messages = get_messages(subject)

    # --- Display Chat History (subject-specific) ---
    for msg in subject_messages:
        with st.chat_message(msg["role"]):
            if "image" in msg:
                st.image(msg["image"], width=350)
            st.markdown(msg["content"], unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # CONFIDENCE CHECK — 3 compact buttons: No / Yes / Crystal clear!
    # "No" (rating 1) auto-triggers a simpler re-explanation.
    # All ratings saved to rating_log for future analytics.
    # -------------------------------------------------------------------------
    if st.session_state.show_confidence:
        st.markdown(
            "<p style='margin:6px 0 4px;font-size:0.85rem;font-weight:600;"
            "color:var(--text-color)'>Did you understand this?</p>",
            unsafe_allow_html=True
        )
        # 3 tight buttons + 1 wide spacer to left-anchor them
        # CONFIDENCE_OPTIONS imported from config.py
        c_cols = st.columns([2, 2, 3, 5])
        for i, (label, rating, should_reexplain) in enumerate(CONFIDENCE_OPTIONS):
            with c_cols[i]:
                if st.button(label, key=f"conf_{i}_{subject}", use_container_width=True):
                    st.session_state.rating_log.append({
                        "student": st.session_state.student_name,
                        "subject": subject,
                        "grade":   st.session_state.grade,
                        "rating":  rating,
                        "label":   label
                    })
                    st.session_state.show_confidence = False
                    if should_reexplain:
                        st.session_state.pending_prompt = (
                            "I didn't understand that well. "
                            "Can you explain it again in a simpler way with a different example?"
                        )
                    st.rerun()

    # -------------------------------------------------------------------------
    # PRACTICE QUESTION + FOLLOW-UPS — All in one compact inline row
    # Practice button on the left, follow-up suggestions next to it.
    # Number of columns = 1 (practice) + number of follow-ups + 1 (spacer).
    # -------------------------------------------------------------------------
    show_pq = bool(st.session_state.practice_q)
    show_fu = bool(st.session_state.followups)

    if show_pq or show_fu:
        st.markdown(
            "<p style='margin:8px 0 4px;font-size:0.85rem;font-weight:600;"
            "color:var(--text-color)'>💡 What next?</p>",
            unsafe_allow_html=True
        )
        # Build column list: each button gets weight 2, spacer gets the rest
        fu_count  = len(st.session_state.followups) if show_fu else 0
        pq_count  = 1 if show_pq else 0
        btn_count = pq_count + fu_count
        # Tight weights for buttons, large spacer at end to left-anchor them
        col_weights = [2] * btn_count + [max(1, 12 - btn_count * 2)]
        btn_cols = st.columns(col_weights)

        col_idx = 0

        # Practice question button
        if show_pq:
            with btn_cols[col_idx]:
                if st.button(
                    "🎯 Practice",
                    key=f"pq_btn_{subject}",
                    help=st.session_state.practice_q,
                    use_container_width=True
                ):
                    pq = st.session_state.practice_q
                    st.session_state.practice_q = ""
                    st.session_state.followups  = []
                    st.session_state.show_confidence = False
                    st.session_state.pending_prompt = (
                        f"Give me this practice question and wait for my answer before explaining: {pq}"
                    )
                    st.rerun()
            col_idx += 1

        # Follow-up suggestion buttons
        if show_fu:
            for i, suggestion in enumerate(st.session_state.followups):
                with btn_cols[col_idx]:
                    if st.button(
                        suggestion,
                        key=f"followup_{i}_{subject}",
                        use_container_width=True
                    ):
                        st.session_state.pending_prompt = suggestion
                        st.session_state.followups = []
                        st.rerun()
                col_idx += 1

    # --- Staged Image Preview ---
    if st.session_state.staged_image and not any(
        m.get("image_id") == st.session_state.last_uploaded
        for m in subject_messages
    ):
        st.info("🖼️ Image staged and ready. Type your question and press Enter.")
        st.image(st.session_state.staged_image, width=350)

    # --- Chat Input ---
    # Accepts input either from the text box OR a follow-up button click
    typed_prompt = st.chat_input("Type your doubt here...")
    prompt = typed_prompt or st.session_state.pop("pending_prompt", None)

    if prompt:
        # Clear follow-up suggestions whenever a new question comes in
        st.session_state.followups = []
        user_msg = {"role": "user", "content": prompt}

        # =====================================================================
        # PATH A: Image + Text → Vision Model (with streaming)
        # =====================================================================
        if st.session_state.staged_image:
            # --- Option 2: Subject pre-check before hitting the vision model ---
            chat_history_for_check = str(get_memory(subject).chat_memory.messages)
            if not is_question_on_subject(prompt, subject, chat_history_for_check, chat_ai):
                subject_messages.append(user_msg)
                with st.chat_message("user"):
                    st.image(st.session_state.staged_image, width=350)
                    st.markdown(prompt)
                refusal = build_off_subject_message(subject, prompt, chat_ai)
                with st.chat_message("assistant"):
                    st.markdown(refusal)
                subject_messages.append({"role": "assistant", "content": refusal})
                st.session_state.staged_image = None
                st.rerun()

            user_msg["image"] = st.session_state.staged_image
            user_msg["image_id"] = st.session_state.last_uploaded
            subject_messages.append(user_msg)

            with st.chat_message("user"):
                st.image(st.session_state.staged_image, width=350)
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    b64 = encode_image(st.session_state.staged_image)

                    vision_system = build_system_prompt(
                        st.session_state.grade, subject, hint_mode,
                        st.session_state.student_name
                    ).replace(
                        "{context}",
                        "[No textbook context for image questions. Use general knowledge.]"
                    )

                    # Pass subject-scoped memory history to vision model
                    vision_messages = [SystemMessage(content=vision_system)]
                    for mem_msg in get_memory(subject).chat_memory.messages:
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

                    # Stream the vision model response
                    answer = st.write_stream(vision_ai.stream(vision_messages))

                    # Save to subject memory manually
                    get_memory(subject).chat_memory.add_user_message(prompt)
                    get_memory(subject).chat_memory.add_ai_message(answer)

                    subject_messages.append({"role": "assistant", "content": answer})
                    st.session_state.staged_image = None

                    # Image questions always use general knowledge (no RAG) —
                    # show fallback warning so student knows this
                    st.markdown(
                        "<div class='fallback-warning'>⚠️ This answer is based on general knowledge, "
                        "not your specific textbook. Please verify with your teacher or notes.</div>",
                        unsafe_allow_html=True
                    )

                    # Run both post-answer LLM calls in parallel (faster than sequential)
                    with st.spinner("Generating suggestions..."):
                        fu, pq = generate_post_answer_content(
                            answer, subject, st.session_state.grade
                        )
                    st.session_state.followups  = fu
                    st.session_state.practice_q = pq
                    st.session_state.show_confidence = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

        # =====================================================================
        # PATH B: Text-only → Streaming RAG answer + follow-up suggestions
        # =====================================================================
        else:
            # --- Option 2: Subject pre-check before hitting the RAG chain ---
            chat_history_for_check = str(get_memory(subject).chat_memory.messages)
            if not is_question_on_subject(prompt, subject, chat_history_for_check, chat_ai):
                subject_messages.append(user_msg)
                with st.chat_message("user"):
                    st.markdown(prompt)
                refusal = build_off_subject_message(subject, prompt, chat_ai)
                with st.chat_message("assistant"):
                    st.markdown(refusal)
                subject_messages.append({"role": "assistant", "content": refusal})
                st.rerun()

            subject_messages.append(user_msg)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # Stream answer — returns (answer, docs, used_fallback)
                    # Single retrieval call covers both answer + source tags
                    answer, source_docs, used_fallback = get_streaming_answer(
                        qa_chain, prompt,
                        st.session_state.grade, subject, hint_mode
                    )

                    # Show source tags using docs already retrieved — no second retrieval
                    if source_docs:
                        sources = set([
                            os.path.basename(d.metadata.get('source', 'Unknown'))
                            for d in source_docs
                        ])
                        source_tag = f"\n\n<span class='source-tag'>📚 Source: {', '.join(sources)}</span>"
                        st.markdown(source_tag, unsafe_allow_html=True)
                        answer += source_tag

                    # Fallback warning — shown when RAG found nothing in the textbook
                    if used_fallback:
                        st.markdown(
                            "<div class='fallback-warning'>⚠️ I couldn't find this topic in your "
                            "textbook. This answer is from general knowledge — please verify "
                            "with your teacher or notes.</div>",
                            unsafe_allow_html=True
                        )

                    subject_messages.append({"role": "assistant", "content": answer})

                    # Run both post-answer LLM calls in parallel (faster than sequential)
                    with st.spinner("Generating suggestions..."):
                        fu, pq = generate_post_answer_content(
                            answer, subject, st.session_state.grade
                        )
                    st.session_state.followups  = fu
                    st.session_state.practice_q = pq
                    st.session_state.show_confidence = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error fetching answer: {e}")