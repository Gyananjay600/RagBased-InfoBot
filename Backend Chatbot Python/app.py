import os
import sys

# Ensure UTF-8 output encoding for console logs on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Base directory for resolving relative paths reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

genai_api_key = os.getenv("GOOGLE_API_KEY")
if not genai_api_key:
    print("WARNING: GOOGLE_API_KEY is not set. Please add it to your .env file.")

# Model configuration
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Folder paths
pdf_folder_path = os.path.join(BASE_DIR, "documents")
faiss_index_path = os.path.join(BASE_DIR, "faiss_index")

app = FastAPI(title="Info-Bot API", description="RAG Based AI Assistant for College Website")

# Add CORS middleware to support all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pdf_text_from_folder(folder_path: str) -> str:
    """Extract and normalize text from all PDF files in the specified folder."""
    text = ""
    if not os.path.exists(folder_path):
        print(f"Warning: PDF directory '{folder_path}' does not exist.")
        return text

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                pdf_reader = PdfReader(pdf_path)
                print(f"Reading '{filename}' ({len(pdf_reader.pages)} pages)...")
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        # Clean single-line breaks that PyPDF2 produces while preserving paragraphs
                        lines = [line.strip() for line in extracted.splitlines() if line.strip()]
                        text += " ".join(lines) + "\n\n"
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
    return text


def get_text_chunks(raw_text: str):
    """Split text into manageable chunks with overlap for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(raw_text)


def build_or_load_vector_store():
    """Load existing FAISS index from disk or build a new one from PDF documents."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=genai_api_key,
    )

    index_file = os.path.join(faiss_index_path, "index.faiss")
    if os.path.exists(index_file):
        try:
            print(f"Loading persistent FAISS vector store from '{faiss_index_path}'...")
            return FAISS.load_local(
                faiss_index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"Failed to load cached index: {e}. Rebuilding...")

    print("Extracting text from documents folder...")
    raw_text = get_pdf_text_from_folder(pdf_folder_path)
    if not raw_text.strip():
        print("Warning: No text extracted from PDFs. Creating empty vector store fallback.")
        raw_text = "No document text available."

    chunks = get_text_chunks(raw_text)
    print(f"Created {len(chunks)} chunks. Generating embeddings with {EMBEDDING_MODEL}...")
    store = FAISS.from_texts(chunks, embedding=embeddings)

    try:
        store.save_local(faiss_index_path)
        print(f"Saved FAISS index to '{faiss_index_path}'.")
    except Exception as e:
        print(f"Warning: Could not save FAISS index: {e}")

    return store


# Initialize vector store on startup
vector_store = None
try:
    vector_store = build_or_load_vector_store()
    print("Vector store initialized successfully.")
except Exception as e:
    print(f"Error initializing vector store: {e}")


def extract_content_text(content) -> str:
    """Normalize response content from ChatGoogleGenerativeAI to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


PROMPT_TEMPLATE = """You are a helpful college assistant who answers questions based on the snippets of text provided in context.
Answer only using the context provided, being as accurate, clear, and helpful as possible.
If the information is not available in the context, respond with:
"The answer is not available in the context provided."

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the FastAPI Info-Bot backend. Send POST requests to /ask to interact.",
        "models": {
            "chat": CHAT_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
        "vector_store_ready": vector_store is not None,
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "vector_store": vector_store is not None}


@app.post("/reindex")
def reindex_documents():
    """Endpoint to rebuild the FAISS vector store from documents/."""
    global vector_store
    try:
        if os.path.exists(faiss_index_path):
            import shutil
            shutil.rmtree(faiss_index_path, ignore_errors=True)
        vector_store = build_or_load_vector_store()
        return {"status": "success", "message": "Documents re-indexed successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to reindex: {str(e)}"})


@app.post("/ask")
async def ask_question(request: Request):
    global vector_store
    try:
        data = await request.json()
        question = data.get("question")
        if not question or not question.strip():
            return JSONResponse(content={"error": "Question not provided"}, status_code=400)

        if vector_store is None:
            # Try lazy initialization if it failed on startup
            try:
                vector_store = build_or_load_vector_store()
            except Exception as e:
                return JSONResponse(
                    content={"error": f"Vector store not ready: {str(e)}"},
                    status_code=503,
                )

        docs = vector_store.similarity_search(question.strip(), k=4)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        model = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0.3,
            google_api_key=genai_api_key,
        )

        chain = prompt | model
        response = chain.invoke({"context": context_text, "question": question.strip()})
        answer_text = extract_content_text(response.content)

        return {"answer": answer_text}
    except Exception as e:
        print(f"Error processing question: {e}")
        return JSONResponse(content={"error": f"Internal server error: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
