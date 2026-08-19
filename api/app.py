import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Load .env file FIRST before reading any os.getenv() calls
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
    print(f"Loaded .env from: {_env_path}")
except ImportError:
    print("Warning: python-dotenv not installed. Falling back to system environment variables.")

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_pipeline import RAGPipeline
from llm.openrouter_client import OpenRouterClient

app = FastAPI(
    title="TokenWise API",
    description="Smart Context Compression for RAG - Production REST API",
    version="1.0.0"
)

# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173")
origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Global Pipeline Instance (Lazy Loaded)
# ---------------------------------------------------------
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

_pipeline_instance: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        print("Initializing TokenWise RAGPipeline & loading ML models...")
        _pipeline_instance = RAGPipeline(
            documents_dir=DOCUMENTS_DIR,
            chunk_size=3,
            chunk_overlap=1,
            top_k=int(os.getenv("DEFAULT_TOP_K", "4")),
            token_budget=int(os.getenv("DEFAULT_TOKEN_BUDGET", "80")),
            coverage_threshold=float(os.getenv("DEFAULT_COVERAGE_THRESHOLD", "0.80"))
        )
        try:
            _pipeline_instance.build_index()
        except Exception as e:
            print(f"Warning during initial index build: {e}")
    return _pipeline_instance

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query to process")
    top_k: Optional[int] = Field(default=4, ge=1, le=20)
    token_budget: Optional[int] = Field(default=80, ge=16, le=2048)
    coverage_threshold: Optional[float] = Field(default=0.80, ge=0.1, le=1.0)


class BenchmarkRequest(BaseModel):
    query: str = Field(..., description="User query to benchmark")
    top_k: Optional[int] = Field(default=4, ge=1, le=20)
    token_budget: Optional[int] = Field(default=80, ge=16, le=2048)
    coverage_threshold: Optional[float] = Field(default=0.80, ge=0.1, le=1.0)
    runs: Optional[int] = Field(default=5, ge=1, le=10)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check endpoint - returns immediately without blocking."""
    is_loaded = _pipeline_instance is not None
    return {
        "status": "ok",
        "is_initialized": is_loaded,
        "is_indexed": _pipeline_instance.is_indexed if is_loaded else False,
        "indexed_chunks": _pipeline_instance.retriever.document_count if is_loaded else 0,
        "documents_dir": DOCUMENTS_DIR
    }


@app.get("/api/documents")
def list_documents():
    """List all available documents in the ingestion workspace."""
    doc_dir = Path(DOCUMENTS_DIR)
    if not doc_dir.exists():
        return {"documents": []}

    supported_exts = {".txt", ".pdf", ".docx", ".md"}
    docs_info = []

    pipe = get_pipeline()

    # Map chunk counts per document filename
    chunk_counts = {}
    for chunk in pipe.chunks:
        doc_name = chunk.get("document_name")
        if doc_name:
            chunk_counts[doc_name] = chunk_counts.get(doc_name, 0) + 1

    for path in sorted(doc_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_exts:
            stat = path.stat()
            docs_info.append({
                "filename": path.name,
                "file_type": path.suffix.lower().replace(".", "").upper(),
                "size_bytes": stat.st_size,
                "size_formatted": f"{stat.st_size / 1024:.1f} KB" if stat.st_size >= 1024 else f"{stat.st_size} B",
                "chunk_count": chunk_counts.get(path.name, 0),
                "modified_time": int(stat.st_mtime)
            })

    return {
        "documents": docs_info,
        "total_documents": len(docs_info),
        "total_chunks": pipe.retriever.document_count,
        "is_indexed": pipe.is_indexed
    }


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents (.txt, .pdf, .docx, .md)."""
    supported_exts = {".txt", ".pdf", ".docx", ".md"}
    saved_files = []
    errors = []

    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in supported_exts:
            errors.append(f"Unsupported file type '{file_ext}' for file {file.filename}. Supported: .txt, .pdf, .docx, .md")
            continue

        target_path = Path(DOCUMENTS_DIR) / file.filename
        try:
            content = await file.read()
            target_path.write_bytes(content)
            saved_files.append(file.filename)
        except Exception as e:
            errors.append(f"Failed to save {file.filename}: {str(e)}")

    return {
        "message": f"Successfully uploaded {len(saved_files)} file(s). Click 'ANALYZE DOCUMENT' to build FAISS index.",
        "saved_files": saved_files,
        "errors": errors
    }


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Delete a document and update FAISS retrieval index."""
    safe_filename = Path(filename).name
    target_path = Path(DOCUMENTS_DIR) / safe_filename

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found.")

    try:
        target_path.unlink()
        pipe = get_pipeline()
        indexing_summary = pipe.build_index()
        return {
            "message": f"Document '{safe_filename}' deleted successfully.",
            "indexing_summary": indexing_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.post("/api/documents/index")
def rebuild_index():
    """Trigger manual index rebuild."""
    try:
        pipe = get_pipeline()
        summary = pipe.build_index()
        return {
            "message": "Index rebuilt successfully.",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.post("/api/query")
def run_query(request: QueryRequest):
    """Execute complete Retrieve -> Rank -> Compress -> Coverage Guard -> OpenRouter LLM pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    pipe = get_pipeline()

    if not pipe.is_indexed or pipe.retriever.document_count == 0:
        # Try to build index if documents exist
        doc_files = [p for p in Path(DOCUMENTS_DIR).glob("*") if p.is_file()] if Path(DOCUMENTS_DIR).exists() else []
        if doc_files:
            pipe.build_index()
        else:
            raise HTTPException(
                status_code=400,
                detail="No documents are currently indexed. Please upload at least one document first and click 'ANALYZE DOCUMENT'."
            )

    # 1. Retrieval
    retrieval_start = time.perf_counter()
    try:
        retrieved_chunks = pipe.retrieve(query=request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    retrieval_time_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

    # 2. Compression & Coverage Guard
    compression_start = time.perf_counter()
    try:
        compression_result = pipe.compress_retrieved(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            token_budget=request.token_budget
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context compression failed: {str(e)}")
    compression_time_ms = round((time.perf_counter() - compression_start) * 1000, 2)

    total_preprocessing_time_ms = round(retrieval_time_ms + compression_time_ms, 2)

    # 3. OpenRouter LLM Answer Generation
    llm_answer = None
    llm_error = None
    llm_latency_ms = 0.0
    llm_prompt_tokens = 0
    llm_completion_tokens = 0
    llm_total_tokens = 0

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        try:
            llm_client = OpenRouterClient(api_key=openrouter_key)
            llm_res = llm_client.generate(
                query=request.query,
                context=compression_result["compressed_context"]
            )
            llm_answer = llm_res["response"]
            llm_latency_ms = llm_res["latency_ms"]
            llm_prompt_tokens = llm_res["prompt_tokens"]
            llm_completion_tokens = llm_res["completion_tokens"]
            llm_total_tokens = llm_res["total_tokens"]
        except Exception as e:
            llm_error = str(e)
    else:
        llm_error = "OPENROUTER_API_KEY is not set in backend environment variables."

    # Return full structured TokenWise output
    return {
        "query": request.query,
        "retrieved_chunks": compression_result.get("retrieved_chunks", []),
        "retrieved_chunk_count": compression_result.get("retrieved_chunk_count", 0),
        "retrieved_context": compression_result.get("retrieved_context", ""),
        "compressed_context": compression_result.get("compressed_context", ""),
        "original_tokens": compression_result.get("original_tokens", 0),
        "compressed_tokens": compression_result.get("compressed_tokens", 0),
        "tokens_saved": compression_result.get("tokens_saved", 0),
        "compression_ratio": round(compression_result.get("compression_ratio", 0.0), 2),
        "coverage": round(compression_result.get("coverage", 0.0), 4),
        "coverage_threshold": request.coverage_threshold,
        "coverage_guard_passed": compression_result.get("coverage_guard_passed", True),
        "coverage_guard_triggered": compression_result.get("coverage_guard_triggered", False),
        "matched_concepts": compression_result.get("matched_concepts", []),
        "missing_concepts": compression_result.get("missing_concepts", []),
        "kept": compression_result.get("kept", []),
        "removed": compression_result.get("removed", []),
        "total_sentences": compression_result.get("total_sentences", 0),
        "kept_sentences": compression_result.get("kept_sentences", 0),
        "removed_sentences": compression_result.get("removed_sentences", 0),
        "token_budget": request.token_budget,
        "retrieval_time_ms": retrieval_time_ms,
        "compression_time_ms": compression_time_ms,
        "total_preprocessing_time_ms": total_preprocessing_time_ms,
        "llm_answer": llm_answer,
        "llm_error": llm_error,
        "llm_latency_ms": llm_latency_ms,
        "llm_prompt_tokens": llm_prompt_tokens,
        "llm_completion_tokens": llm_completion_tokens,
        "llm_total_tokens": llm_total_tokens
    }


@app.post("/api/benchmark")
def run_benchmark(request: BenchmarkRequest):
    """Run comparative benchmark: Original Context vs TokenWise Compressed Context across N runs."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY environment variable is missing. Benchmarking requires an OpenRouter API key."
        )

    # 1. Run pipeline to get original & compressed contexts
    query_res = run_query(QueryRequest(
        query=request.query,
        top_k=request.top_k,
        token_budget=request.token_budget,
        coverage_threshold=request.coverage_threshold
    ))

    original_context = query_res["retrieved_context"]
    compressed_context = query_res["compressed_context"]

    if not original_context.strip():
        raise HTTPException(status_code=400, detail="Retrieved context is empty. Upload documents before benchmarking.")

    llm_client = OpenRouterClient(api_key=openrouter_key)

    original_runs = []
    compressed_runs = []

    # 2. Benchmark Original Context
    for run_num in range(1, request.runs + 1):
        try:
            res = llm_client.generate(query=request.query, context=original_context)
            res["run"] = run_num
            original_runs.append(res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Benchmark error on original context run {run_num}: {str(e)}")

    # 3. Benchmark Compressed Context
    for run_num in range(1, request.runs + 1):
        try:
            res = llm_client.generate(query=request.query, context=compressed_context)
            res["run"] = run_num
            compressed_runs.append(res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Benchmark error on compressed context run {run_num}: {str(e)}")

    # 4. Latency calculations
    orig_latencies = [r["latency_ms"] for r in original_runs]
    comp_latencies = [r["latency_ms"] for r in compressed_runs]

    orig_avg_lat = sum(orig_latencies) / len(orig_latencies)
    comp_avg_lat = sum(comp_latencies) / len(comp_latencies)

    lat_diff_ms = round(orig_avg_lat - comp_avg_lat, 2)
    lat_reduction_pct = round(((orig_avg_lat - comp_avg_lat) / orig_avg_lat) * 100, 2) if orig_avg_lat > 0 else 0.0

    if lat_reduction_pct >= 5.0:
        latency_label = "Meaningful improvement"
    elif lat_reduction_pct <= -5.0:
        latency_label = "Regression"
    else:
        latency_label = "Marginal / within benchmark variation"

    # 5. Token metrics calculations
    orig_prompt_tokens = original_runs[0]["prompt_tokens"]
    comp_prompt_tokens = compressed_runs[0]["prompt_tokens"]
    prompt_tokens_saved = orig_prompt_tokens - comp_prompt_tokens
    prompt_token_reduction_pct = round(((orig_prompt_tokens - comp_prompt_tokens) / orig_prompt_tokens) * 100, 2) if orig_prompt_tokens > 0 else 0.0

    orig_comp_tokens_avg = sum(r["completion_tokens"] for r in original_runs) / len(original_runs)
    comp_comp_tokens_avg = sum(r["completion_tokens"] for r in compressed_runs) / len(compressed_runs)

    orig_total_tokens_avg = orig_prompt_tokens + orig_comp_tokens_avg
    comp_total_tokens_avg = comp_prompt_tokens + comp_comp_tokens_avg
    total_tokens_saved = orig_total_tokens_avg - comp_total_tokens_avg
    total_token_reduction_pct = round(((orig_total_tokens_avg - comp_total_tokens_avg) / orig_total_tokens_avg) * 100, 2) if orig_total_tokens_avg > 0 else 0.0

    # 6. Answer consistency check
    orig_answer = original_runs[0]["response"]
    comp_answer = compressed_runs[0]["response"]

    # Basic text similarity / match verification
    if orig_answer.strip().lower() == comp_answer.strip().lower():
        answer_consistency = "MATCH"
    else:
        # Check concept overlap
        orig_words = set(orig_answer.lower().split())
        comp_words = set(comp_answer.lower().split())
        overlap = len(orig_words.intersection(comp_words)) / max(1, len(orig_words.union(comp_words)))
        answer_consistency = "MATCH" if overlap > 0.4 else "DIFFERENT"

    return {
        "query": request.query,
        "runs": request.runs,
        "original_answer": orig_answer,
        "compressed_answer": comp_answer,
        "answer_consistency": answer_consistency,
        "original_avg_latency_ms": round(orig_avg_lat, 2),
        "compressed_avg_latency_ms": round(comp_avg_lat, 2),
        "latency_diff_ms": lat_diff_ms,
        "latency_reduction_pct": lat_reduction_pct,
        "latency_label": latency_label,
        "original_min_latency_ms": round(min(orig_latencies), 2),
        "original_max_latency_ms": round(max(orig_latencies), 2),
        "compressed_min_latency_ms": round(min(comp_latencies), 2),
        "compressed_max_latency_ms": round(max(comp_latencies), 2),
        "original_prompt_tokens": orig_prompt_tokens,
        "compressed_prompt_tokens": comp_prompt_tokens,
        "prompt_tokens_saved": prompt_tokens_saved,
        "prompt_token_reduction_pct": prompt_token_reduction_pct,
        "original_completion_tokens_avg": round(orig_comp_tokens_avg, 1),
        "compressed_completion_tokens_avg": round(comp_comp_tokens_avg, 1),
        "original_total_tokens_avg": round(orig_total_tokens_avg, 1),
        "compressed_total_tokens_avg": round(comp_total_tokens_avg, 1),
        "total_tokens_saved": round(total_tokens_saved, 1),
        "total_token_reduction_pct": total_token_reduction_pct,
        "original_runs": original_runs,
        "compressed_runs": compressed_runs
    }
