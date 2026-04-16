# EduSolve AI 🎓

An AI-powered tutoring chatbot built for students of Class 8, 9, and 10. I built this for a local coaching class near my house in Pune. Students can ask doubts in Math, Science, and English and get clear step-by-step answers in simple English (with Marathi support too).

---

## What it does

- Answers student doubts based on their actual textbooks using RAG (Retrieval-Augmented Generation)
- Supports image uploads — students can photograph a question and ask about it
- Remembers the last 5 questions per subject so follow-up questions work naturally
- Hint Mode — instead of giving the full answer, the agent guides the student with clues
- Generates follow-up question suggestions and a practice question after every answer
- Confidence check after every answer (👎 / 👍 / 🤩) — if the student doesn't understand, it automatically re-explains in a simpler way
- Answers primarily in English, with optional Marathi clarification
- Separate chat history per subject (Math, Science, English) within a session
- Streaming responses so answers appear word by word

---

## Tech Stack

- **Frontend** — Streamlit
- **LLM** — Llama 3.3 70B via Groq API (text), Llama 4 Scout (vision/image)
- **RAG** — LangChain + ChromaDB + HuggingFace Embeddings (all-MiniLM-L6-v2)
- **PDF Ingestion** — LangChain PyPDFLoader

---

## Project Structure

```
EduSolve/
├── app.py            # Main Streamlit app
├── config.py         # All settings in one place (models, grades, paths)
├── prompts.py        # All LLM prompt functions
├── ingest.py         # Script to load PDFs into the vector database
├── list_models.py    # Utility to check available Groq models
├── requirements.txt
├── .env              # Your GROQ_API_KEY goes here
├── data/             # Put your PDF textbooks here
└── db/               # ChromaDB vector store (auto-generated)
```

---

## How to Run

**1. Clone the repo and install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_key_here
```

**3. Add your PDF textbooks**

Place PDFs in the `data/` folder. Name them like this:
```
8th_math.pdf
9th_science.pdf
10th_english.pdf
```

**4. Build the knowledge base**
```bash
python ingest.py
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## PDF Naming Convention

The file name tells the app which grade and subject the PDF belongs to:

| File name | Grade | Subject |
|---|---|---|
| `8th_math.pdf` | Grade 8 | Math |
| `9th_science.pdf` | Grade 9 | Science |
| `10th_english.pdf` | Grade 10 | English |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (get it free at console.groq.com) |

---

## Notes

- The `db/` folder is auto-generated when you run `ingest.py`. Don't manually edit it.
- If you add new PDFs, re-run `ingest.py` to update the knowledge base.
- To change models, grades, or any settings — edit `config.py` only.
