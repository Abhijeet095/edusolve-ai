# =============================================================================
# prompts.py — EduSolve AI Prompt Library
#
# All LLM prompt-building functions live here.
# To change how the agent teaches, hints, generates follow-ups, or creates
# practice questions — edit this file only. No need to touch app.py.
#
# NOTE: chat_ai is passed as a parameter to generate_* functions to avoid
# circular imports (app.py initialises the model, prompts.py uses it).
# =============================================================================

from langchain_groq import ChatGroq


# =============================================================================
# SYSTEM PROMPT
# Core identity of the tutor. Injected as a SystemMessage into every LLM call.
# =============================================================================
def build_system_prompt(grade: str, subject: str, hint_mode: bool, student_name: str = "") -> str:
    """
    Builds the tutor's full system prompt dynamically.
    - grade      → sets grade-level expectations
    - subject    → activates subject-specific rules
    - hint_mode     → switches between full-answer and Socratic hint mode
    - student_name  → personalises greetings and encouragement (optional)
    """
    if hint_mode:
        teaching_strategy = """
HINT MODE IS ON:
- Do NOT give the full answer directly.
- Guide the student step by step using 2-3 hints or leading questions.
- Ask things like "What formula do you think applies here?" or "What did we learn about this topic?"
- If the student is stuck after 2 hints, you may give a small nudge towards the answer.
- Encourage the student to think first before revealing more.
"""
    else:
        teaching_strategy = """
ANSWER MODE IS ON:
- Provide a clear, complete, step-by-step solution.
- For Math: Show every step on a new line. Use **Step 1:**, **Step 2:** format. Wrap all math expressions in $ symbols for LaTeX rendering (e.g., $a^2 + b^2 = c^2$).
- For Science: First explain the concept simply, then connect it to a real-life example the student can relate to (things they see in daily life, at home, or in Pune/Maharashtra).
- For English: Correct mistakes kindly, explain the grammar rule simply, and give a relatable example.
- End longer answers with a short "📝 Quick Summary:" section.
"""

    student_name_line = student_name if student_name else "unknown (ask them their name if it feels natural)"

    return f"""
You are EduSolve — a friendly, patient, and encouraging AI tutor for students of a local coaching class in Pune, Maharashtra.
You are currently helping a student from {grade} with their {subject} doubts.
The student's name is {student_name_line}. Use their name naturally in your responses — when encouraging them, 
correcting them kindly, or celebrating a good answer. Don't overuse it — 1-2 times per response is enough.

YOUR PERSONALITY:
- Speak like a good teacher — warm, clear, and supportive.
- Never make a student feel bad for not knowing something. Always be encouraging.
- Use phrases like "Good question!", "You're on the right track!", or "Let's figure this out together!" where appropriate.
- Never be rude, dismissive, or overly formal.

SUBJECT RESTRICTION (VERY IMPORTANT — FOLLOW THIS STRICTLY):
- You are configured ONLY for {subject} right now.
- You MUST ONLY answer questions that are clearly related to {subject}.
- If the student asks a question that belongs to a DIFFERENT school subject (for example, they selected English but ask a Maths or Science question), you must REFUSE politely.
- When refusing an off-subject question, say exactly this (fill in the blanks):
  "I'm set up for {subject} right now! To get help with [other subject], please switch the subject using the dropdown in the sidebar. I'm here for all your {subject} doubts! 😊"
- Do NOT attempt to answer even partially if the question is clearly from another subject.
- This restriction applies even if the question seems simple or you know the answer.
- Questions about {subject} topics, concepts, grammar, problems, or examples ARE allowed.
- General greetings, "explain again", "give an example" follow-ups ARE allowed — treat them as {subject} follow-ups based on chat history.

CONVERSATION MEMORY:
- You remember the last few messages of this conversation.
- Use this history to understand follow-up questions like "explain that again", "give another example", or "I didn't understand step 2".
- If the student refers to something from earlier (e.g. "that formula you mentioned"), look at the chat history and respond accordingly.
- Never ask the student to repeat themselves if the answer is already in the conversation history.

LANGUAGE RULES (VERY IMPORTANT):
- Always answer PRIMARILY in English. English must be the main language of every response.
- Use SIMPLE English — avoid complex or rare vocabulary. A Class 8–10 student should understand every word.
- DO NOT use large markdown headers (# or ##) in your response. Use **bold** for labels like **Step 1:** instead.
- If a concept is hard to explain in English alone, you MAY add a short Marathi clarification AFTER the English explanation.
  Format it like this: "थोडक्यात: [one sentence Marathi summary]"
- Never respond ONLY in Marathi unless the student explicitly asks for it.
- Never mix Marathi and English word-by-word in the same sentence. Keep them in separate sections.

WHAT YOU MUST NOT DO:
- Do not answer questions unrelated to studies (e.g., movies, gaming, social media). Politely redirect: "That's outside what I can help with, but I'm here for your {subject} doubts anytime!"
- Do not write full exam papers or complete entire homework assignments. Help the student understand, not just copy.
- Do not use jargon or technical terms without explaining them in simple words first.
- Do not give very long, overwhelming responses. Keep answers focused — a student should read it in 1–2 minutes.

{teaching_strategy}

SUBJECT-SPECIFIC RULES:
- Math: Always show full working. Never skip steps. Wrap all math in $ for LaTeX.
- Science: Use real-life analogies. Connect concepts to things students see in daily life.
- English: Be kind when correcting. Explain the rule, give an example, then show the corrected version.

CONTEXT FROM STUDY MATERIAL:
The following is relevant content retrieved from the student's textbooks and notes.
Use this as your PRIMARY source of information when answering. If the context is sufficient, base your answer on it.
If the context is not enough, use your general knowledge but stay accurate and grade-appropriate.

{{context}}
""".strip()


# =============================================================================
# SUBJECT CLASSIFIER (Option 2 — Pre-check before main LLM call)
# Quickly checks whether the student's question belongs to the selected subject.
# Returns True if the question is on-subject (or a follow-up/greeting), False if off-subject.
# This is the safety net — catches edge cases where the system prompt alone might fail.
# =============================================================================
def is_question_on_subject(question: str, subject: str, chat_history: str, chat_ai: ChatGroq) -> bool:
    """
    Runs a fast YES/NO classifier call to check if the question belongs to `subject`.
    Returns True  → question is relevant to the subject (safe to proceed).
    Returns False → question is from a different school subject (should be blocked).

    chat_history is included so that follow-up questions like "explain that again"
    are correctly classified as on-subject (they refer to the current subject's context).
    chat_ai is passed in to avoid circular imports.
    """
    prompt = f"""You are a strict subject classifier for a school tutor app.

The student has selected: **{subject}**
Recent chat history (for context):
{chat_history if chat_history else "[No previous messages — this is the first question]"}

Student's new question: "{question}"

Your job: Decide if this question belongs to {subject} or is a follow-up to the current {subject} conversation.

Rules:
- Answer YES if the question is about {subject} topics, concepts, problems, or grammar.
- Answer YES if it is a general follow-up like "explain again", "give example", "I didn't understand" — these refer to the current {subject} conversation.
- Answer YES if it is a greeting or small talk (e.g. "hi", "ok", "thanks").
- Answer NO if the question clearly belongs to a DIFFERENT school subject (e.g. student selected English but asks a Maths equation, or selected Maths but asks about photosynthesis).
- When in doubt, answer YES.

Reply with ONLY one word: YES or NO. Nothing else."""

    try:
        res = chat_ai.invoke(prompt)
        answer = res.content.strip().upper()
        # If the response is anything other than a clear NO, treat it as on-subject
        return answer != "NO"
    except Exception as e:
        print(f"[is_question_on_subject] Classifier failed: {e}. Defaulting to YES.")
        # Fail open — if classifier errors out, let the main chain handle it
        return True


# =============================================================================
# OFF-SUBJECT REFUSAL MESSAGE
# Returns a friendly, consistent refusal message when classifier returns False.
# Centralised here so it can be changed in one place.
# =============================================================================
def build_off_subject_message(subject: str, question: str, chat_ai: ChatGroq) -> str:
    """
    Generates a warm refusal message telling the student to switch subjects.
    The message is generated dynamically so it can acknowledge what the student asked.
    chat_ai is passed in to avoid circular imports.
    """
    prompt = f"""A student asked a question that belongs to a DIFFERENT school subject, but they have selected {subject} as their current subject.

Their question was: "{question}"

Write a SHORT, friendly, encouraging refusal message (2-3 sentences max) that:
1. Tells them this question is not for {subject}
2. Suggests they switch the subject using the dropdown in the sidebar
3. Reassures them you are ready for their {subject} doubts

Keep it warm and supportive. Do NOT answer the question at all.
Return ONLY the message text, nothing else."""

    try:
        res = chat_ai.invoke(prompt)
        return res.content.strip()
    except Exception as e:
        print(f"[build_off_subject_message] Failed: {e}. Using default message.")
        return (
            f"I'm set up for **{subject}** right now! 😊 "
            f"It looks like your question is from a different subject. "
            f"Please switch the subject using the dropdown in the sidebar to get help with that. "
            f"I'm here for all your {subject} doubts!"
        )


# =============================================================================
# QUESTION REWRITE PROMPT
# Turns vague follow-ups like "explain that again" into standalone questions
# like "Explain photosynthesis again in simple terms" so RAG retrieval works.
# =============================================================================
def build_rewrite_prompt(chat_history: str, question: str) -> str:
    """Returns a prompt that rewrites a follow-up into a standalone question."""
    return f"""Given this conversation history:
{chat_history}

Rewrite this follow-up question as a complete standalone question:
"{question}"

Return ONLY the rewritten question, nothing else."""


# =============================================================================
# FOLLOW-UP SUGGESTION GENERATOR
# Generates 3 short natural questions a student might want to ask next.
# Shown as clickable buttons below each answer.
# =============================================================================
def generate_followups(answer: str, subject: str, grade: str, chat_ai: ChatGroq) -> list[str]:
    """
    Returns a list of up to 3 short follow-up question strings, or [] on failure.
    chat_ai is passed in to avoid circular imports.
    """
    prompt = f"""A student from {grade} just received this answer in their {subject} class:

{answer}

Generate exactly 3 short follow-up questions the student might want to ask next.
Rules:
- Each question must be under 10 words
- Questions should be natural, like a student would actually ask
- No numbering, no bullet points, no extra text
- Return ONLY the 3 questions, one per line, nothing else

Example format:
Can you give me another example?
What is the formula for this?
How is this used in real life?"""

    try:
        res = chat_ai.invoke(prompt)
        lines = [q.strip() for q in res.content.strip().split("\n") if q.strip()]
        return lines[:3]
    except Exception as e:
        print(f"[generate_followups] Failed: {e}")
        return []


# =============================================================================
# PRACTICE QUESTION GENERATOR
# Generates one fresh exam-style question on the same topic as the last answer.
# Shown as a clickable "🎯 Practice" button below each answer.
# =============================================================================
def generate_practice_question(answer: str, subject: str, grade: str, chat_ai: ChatGroq) -> str:
    """
    Returns one practice question as a plain string, or "" on failure.
    chat_ai is passed in to avoid circular imports.
    """
    prompt = f"""A {grade} student just learned about this topic in {subject}:

{answer}

Create exactly ONE practice question on this topic suitable for a {grade} exam.
Rules:
- The question must test understanding, not just memorization
- It must be different from anything already explained above
- For Math: include numbers and ask for a calculation or proof
- For Science: ask for an explanation or application of a concept
- For English: give a sentence to correct, or ask to use a word/rule in context
- Keep it concise — one clear question only
- Return ONLY the question, nothing else. No 'Question:', no numbering."""

    try:
        res = chat_ai.invoke(prompt)
        return res.content.strip()
    except Exception as e:
        print(f"[generate_practice_question] Failed: {e}")
        return ""