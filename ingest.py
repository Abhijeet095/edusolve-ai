import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    EMBEDDING_MODEL,
    DB_PATH,
    DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GRADES,
    SUBJECTS
)

load_dotenv()

# Build lookup sets for fast validation
# e.g. VALID_GRADES = {"Grade_8", "Grade_9", "Grade_10"}
# e.g. VALID_SUBJECTS = {"Math", "Science", "English"}
VALID_GRADES   = set(GRADES)
VALID_SUBJECTS = set(SUBJECTS)


def build_knowledge_base():
    all_chunks = []

    # -------------------------------------------------------------------------
    # DUPLICATE GUARD
    # If the DB folder already exists and has content, stop immediately.
    # This prevents re-running ingest.py from duplicating all chunks in Chroma,
    # which degrades retrieval quality over time.
    # To re-ingest: delete the db/ folder manually, then run ingest.py again.
    # -------------------------------------------------------------------------
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("⚠️  Knowledge base already exists at:", DB_PATH)
        print("   To re-ingest, delete the db/ folder first, then run ingest.py again.")
        print("   Skipping ingestion to prevent duplicate chunks.")
        return

    print("🚀 Starting Local Ingestion...")

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if not file.endswith(".pdf"):
                continue

            print(f"📂 Processing: {file}")

            # -----------------------------------------------------------------
            # FILENAME VALIDATION
            # Expected convention: {grade_number}th_{subject}.pdf
            # e.g. 8th_math.pdf, 10th_science.pdf, 9th_english.pdf
            # -----------------------------------------------------------------
            file_parts = file.replace(".pdf", "").split("_")
            if len(file_parts) < 2:
                print(f"  ⚠️  Skipping {file} — filename must be like 8th_math.pdf")
                continue

            grade_metadata   = f"Grade_{file_parts[0].replace('th', '')}"
            subject_metadata = file_parts[1].capitalize()

            # Validate grade against config.py GRADES list
            if grade_metadata not in VALID_GRADES:
                print(
                    f"  ⚠️  Skipping {file} — grade '{grade_metadata}' is not in GRADES config.\n"
                    f"     Valid grades: {sorted(VALID_GRADES)}\n"
                    f"     Check your filename or add this grade to config.py."
                )
                continue

            # Validate subject against config.py SUBJECTS list
            if subject_metadata not in VALID_SUBJECTS:
                print(
                    f"  ⚠️  Skipping {file} — subject '{subject_metadata}' is not in SUBJECTS config.\n"
                    f"     Valid subjects: {sorted(VALID_SUBJECTS)}\n"
                    f"     Check your filename (e.g. use 'math' not 'maths') or add this subject to config.py."
                )
                continue

            print(f"  ✅ Valid: {grade_metadata} / {subject_metadata}")

            loader = PyPDFLoader(os.path.join(root, file))
            docs   = loader.load()

            for doc in docs:
                doc.metadata["grade"]   = grade_metadata
                doc.metadata["subject"] = subject_metadata

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"  📄 {len(chunks)} chunks created from {file}")

    if not all_chunks:
        print("\n❌ No valid PDF files found.")
        print("   Make sure your PDFs are in the data/ folder and named correctly.")
        print(f"   Valid grades:   {sorted(VALID_GRADES)}")
        print(f"   Valid subjects: {sorted(VALID_SUBJECTS)}")
        print("   Example filenames: 8th_math.pdf, 10th_science.pdf, 9th_english.pdf")
        return

    print(f"\n⏳ Generating embeddings for {len(all_chunks)} chunks (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✅ Knowledge base built successfully!")
    print(f"   Location: {DB_PATH}")
    print(f"   Total chunks stored: {len(all_chunks)}")


if __name__ == "__main__":
    build_knowledge_base()