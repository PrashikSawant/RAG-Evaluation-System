# 📊 Day 15 — RAG Evaluation System

Automatically evaluate RAG pipeline quality using 4 metrics —
faithfulness, answer relevance, context precision, and context recall.
Built with Groq, ChromaDB, Sentence Transformers, and Streamlit.

## 💡 What It Does
- Upload PDF or TXT documents
- Ask single questions and get quality scores
- Run batch evaluation across multiple questions
- Scores every answer on 4 RAGAS-inspired metrics
- Color coded results — green, yellow, red
- Per question breakdown with reasons for each score
- Average scores across entire test suite

## 🛠️ Tech Stack
- Python 3.10+
- ChromaDB — persistent vector database
- Sentence Transformers (all-MiniLM-L6-v2) — embeddings
- LangChain Text Splitters — recursive chunking
- PyMuPDF — PDF text extraction
- Groq API (LLaMA 3.3 70B) — answer generation + evaluation
- Streamlit — web interface
- python-dotenv — API key management

## 📊 Evaluation Metrics
- Faithfulness — is the answer grounded in context?
- Answer Relevance — does it address the question?
- Context Precision — were the right chunks retrieved?
- Context Recall — did context have enough information?

## 🚀 Setup & Run

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/day15-rag-evaluation
cd day15-rag-evaluation

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your API key
Create a .env file:
GROQ_API_KEY=your_key_here

### 4. Run
streamlit run app.py

## 📁 Project Structure
day15-rag-evaluation/
├── app.py                # Streamlit UI, 3 tabs
├── rag_engine.py         # RAG pipeline
├── evaluator.py          # 4 evaluation metrics
├── test_questions.py     # Default test questions
├── requirements.txt      # Dependencies
├── .env                  # API key (not committed)
└── .gitignore

## 🔗 Part of 30-Day AI Engineering Bootcamp
Day 15 of 30 | RAG & Vector Databases Phase
