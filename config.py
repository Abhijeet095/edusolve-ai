# =============================================================================
# config.py — EduSolve AI Central Configuration
#
# ALL hardcoded values live here. To change a model, add a grade, adjust
# memory size, or update any setting — edit ONLY this file.
# No need to touch app.py or ingest.py for configuration changes.
# =============================================================================

import os

# Base directory — resolves to the folder where this file lives.
# All other paths are built relative to this, so the app works correctly
# regardless of the working directory at runtime (local, Streamlit Cloud, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# GROQ MODEL SETTINGS
# -----------------------------------------------------------------------------

# Main chat model — used for answering questions, follow-ups, practice questions
CHAT_MODEL        = "llama-3.3-70b-versatile"
CHAT_TEMPERATURE  = 0.1          # Low = more focused, factual answers

# Vision model — used when student uploads an image/photo of a question
VISION_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_TEMPERATURE = 0.3         # Slightly higher for image interpretation


# -----------------------------------------------------------------------------
# EMBEDDING MODEL
# Used by both app.py (for retrieval) and ingest.py (for building the DB)
# IMPORTANT: Must always be the same model in both files
# -----------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------------------------------------------------------
# PATHS (absolute — safe for all deployment platforms)
# -----------------------------------------------------------------------------
DB_PATH   = os.path.join(BASE_DIR, "db")      # Where Chroma stores the vector DB  (ingest.py writes, app.py reads)
DATA_PATH = os.path.join(BASE_DIR, "data")    # Where PDF textbooks are placed for ingestion


# -----------------------------------------------------------------------------
# RAG / RETRIEVAL
# -----------------------------------------------------------------------------
RETRIEVER_K   = 4     # Number of document chunks retrieved per question
CHUNK_SIZE    = 1000  # Characters per chunk when splitting PDFs
CHUNK_OVERLAP = 100   # Overlap between consecutive chunks


# -----------------------------------------------------------------------------
# MEMORY
# Number of past human+AI exchange pairs remembered per subject per session
# k=5 → last 5 questions + 5 answers kept in context
# Increase for longer memory, decrease to save tokens / speed up responses
# -----------------------------------------------------------------------------
MEMORY_WINDOW_K = 5


# -----------------------------------------------------------------------------
# GRADES & SUBJECTS
# Add or remove entries here — the rest of the app updates automatically.
# ingest.py validates PDF filenames against these lists before ingesting.
# -----------------------------------------------------------------------------
GRADES   = ["Grade_8", "Grade_9", "Grade_10"]
SUBJECTS = ["Math", "Science", "English"]


# -----------------------------------------------------------------------------
# CONFIDENCE CHECK BUTTONS
# Each entry: (button label, numeric rating saved to log, trigger re-explain?)
# "No" (rating 1) triggers an automatic simpler re-explanation from the agent
# -----------------------------------------------------------------------------
CONFIDENCE_OPTIONS = [
    ("👎 No",             1, True),
    ("👍 Yes",            3, False),
    ("🤩 Crystal clear!", 5, False),
]


# -----------------------------------------------------------------------------
# APP UI
# -----------------------------------------------------------------------------
APP_TITLE    = "EduSolve AI: Your Personal Tutor"
APP_ICON     = "🎓"
APP_LAYOUT   = "wide"
WELCOME_TEXT = "Welcome! Please select your standard to get started."