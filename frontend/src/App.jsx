import React, { useState, useEffect } from 'react';
import {
  FileText,
  Upload,
  Trash2,
  RefreshCw,
  Play,
  BarChart2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Zap,
  Sliders,
  Layers,
  Database,
  Search,
  Cpu,
  Info,
  Clock,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  FileCheck
} from 'lucide-react';

export default function App() {
  // Config state
  const [topK, setTopK] = useState(4);
  const [tokenBudget, setTokenBudget] = useState(80);
  const [coverageThreshold, setCoverageThreshold] = useState(0.80);
  const [benchmarkRuns, setBenchmarkRuns] = useState(5);

  // App data state
  // App data state
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');

  // Execution state
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzingDoc, setIsAnalyzingDoc] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Result states
  const [queryResult, setQueryResult] = useState(null);
  const [benchmarkResult, setBenchmarkResult] = useState(null);

  // Expandable UI states
  const [expandedChunks, setExpandedChunks] = useState({});
  const [expandedSentences, setExpandedSentences] = useState({});
  const [showOriginalContext, setShowOriginalContext] = useState(false);

  // Robust API fetch with direct fallback
  const apiFetch = async (path, options = {}) => {
    try {
      const res = await fetch(path, options);
      return res;
    } catch (err) {
      if (path.startsWith('/api/')) {
        const directUrl = `http://127.0.0.1:8001${path}`;
        try {
          const resDirect = await fetch(directUrl, options);
          return resDirect;
        } catch (err2) {
          throw err;
        }
      }
      throw err;
    }
  };

  // Helper for API error formatting
  const handleApiError = (err, fallbackText) => {
    if (err.message === 'Failed to fetch' || err.name === 'TypeError') {
      setErrorMsg('Cannot connect to TokenWise backend server. Please start the backend with: uvicorn api.app:app --host 127.0.0.1 --port 8001 --reload');
    } else {
      setErrorMsg(err.message || fallbackText);
    }
  };

  // ---------------------------------------------------------
  // Initial Data Fetching & Polling
  // ---------------------------------------------------------
  const fetchHealth = async () => {
    try {
      const res = await apiFetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
        setErrorMsg((prev) => (prev && prev.includes('Cannot connect to TokenWise backend') ? null : prev));
      }
    } catch (err) {
      console.error('Health fetch error:', err);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await apiFetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Documents fetch error:', err);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchDocuments();
    const interval = setInterval(() => {
      fetchHealth();
      fetchDocuments();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // ---------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------
  const handleFileUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const res = await apiFetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      if (data.errors && data.errors.length > 0) {
        setErrorMsg(data.errors.join(' | '));
      } else {
        setSuccessMsg(`Document uploaded successfully. Click 'ANALYZE & INDEX DOCUMENT' to process chunks into FAISS.`);
      }

      await fetchDocuments();
      await fetchHealth();
    } catch (err) {
      handleApiError(err, 'Upload failed.');
    } finally {
      setIsUploading(false);
      e.target.value = null;
    }
  };

  const handleDeleteDocument = async (filename) => {
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const res = await apiFetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Delete failed');
      }
      setSuccessMsg(`Document '${filename}' deleted.`);
      await fetchDocuments();
      await fetchHealth();
    } catch (err) {
      handleApiError(err, 'Delete failed.');
    }
  };

  const handleAnalyzeAndIndex = async () => {
    setIsAnalyzingDoc(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const res = await apiFetch('/api/documents/index', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Document analysis & index build failed.');
      }

      const indexedCount = data.summary?.indexed || 0;
      setSuccessMsg(`Document analysis complete! Generated & stored ${indexedCount} sentence chunks in FAISS index.`);
      await fetchDocuments();
      await fetchHealth();
    } catch (err) {
      handleApiError(err, 'Document analysis failed.');
    } finally {
      setIsAnalyzingDoc(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!query.trim()) {
      setErrorMsg('Please enter your question in the query input box.');
      return;
    }

    setIsQuerying(true);
    setErrorMsg(null);
    setQueryResult(null);

    try {
      const res = await apiFetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: topK,
          token_budget: tokenBudget,
          coverage_threshold: coverageThreshold,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Query execution failed.');
      }

      setQueryResult(data);
    } catch (err) {
      handleApiError(err, 'Query execution failed.');
    } finally {
      setIsQuerying(false);
    }
  };

  const handleRunBenchmark = async () => {
    if (!query.trim()) {
      setErrorMsg('Please enter your question in the query input box.');
      return;
    }

    setIsBenchmarking(true);
    setErrorMsg(null);
    setBenchmarkResult(null);

    try {
      const res = await apiFetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: topK,
          token_budget: tokenBudget,
          coverage_threshold: coverageThreshold,
          runs: benchmarkRuns,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Benchmark execution failed.');
      }

      setBenchmarkResult(data);
    } catch (err) {
      handleApiError(err, 'Benchmark execution failed.');
    } finally {
      setIsBenchmarking(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#08090d] text-slate-200 flex flex-col font-sans">
      {/* --------------------------------------------------------- */}
      {/* HEADER */}
      {/* --------------------------------------------------------- */}
      <header className="border-b border-[#1f2434] bg-[#0c0e15] px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-900/30 border border-blue-700/40 rounded-lg text-blue-400">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">TokenWise</h1>
                <span className="text-xs px-2 py-0.5 rounded border border-blue-500/30 bg-blue-950/40 text-blue-300 font-mono">
                  v1.0.0
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Smart Context Compression & Evaluation Layer for RAG
              </p>
            </div>
          </div>

          {/* System Status Indicators */}
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#141824] border border-[#22283a]">
              <span className="text-slate-400">FAISS Index:</span>
              <span className={health?.is_indexed ? 'text-emerald-400 font-semibold' : 'text-amber-400 font-semibold'}>
                {health?.is_indexed ? `${health.indexed_chunks} Chunks` : 'Not Indexed'}
              </span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#141824] border border-[#22283a]">
              <span className="text-slate-400">Model:</span>
              <span className="text-slate-200">all-MiniLM-L6-v2</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#141824] border border-[#22283a]">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              <span className="text-emerald-400 font-medium">Backend Ready</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 space-y-8">
        {/* Error Notification */}
        {errorMsg && (
          <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-200 flex items-start gap-3 text-sm">
            <XCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold text-rose-300">System Alert: </span>
              {errorMsg}
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-rose-400 hover:text-white text-xs font-mono"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* Success Notification */}
        {successMsg && (
          <div className="p-4 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-emerald-200 flex items-start gap-3 text-sm">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold text-emerald-300">Success: </span>
              {successMsg}
            </div>
            <button
              onClick={() => setSuccessMsg(null)}
              className="text-emerald-400 hover:text-white text-xs font-mono"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* --------------------------------------------------------- */}
        {/* TOP SECTION: WORKSPACE & CONFIG */}
        {/* --------------------------------------------------------- */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Document Upload & Management (2 cols) */}
          <div className="lg:col-span-2 bg-[#10121a] border border-[#1f2434] rounded-xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-blue-400" />
                  <h2 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                    Document Workspace
                  </h2>
                </div>
                <button
                  onClick={handleAnalyzeAndIndex}
                  disabled={isAnalyzingDoc || documents.length === 0}
                  className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded bg-[#181c2b] border border-[#262c42] hover:bg-[#20263a] disabled:opacity-50 text-slate-300 transition-colors"
                  title="Rebuild Retrieval Index"
                >
                  <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${isAnalyzingDoc && 'animate-spin'}`} />
                  Rebuild Index
                </button>
              </div>

              {/* Upload Dropzone */}
              <label className="border-2 border-dashed border-[#23293c] hover:border-blue-500/50 bg-[#0c0e15] rounded-lg p-5 flex flex-col items-center justify-center cursor-pointer transition-colors group mb-4">
                <Upload className="w-6 h-6 text-slate-400 group-hover:text-blue-400 mb-2 transition-colors" />
                <span className="text-xs font-medium text-slate-300">
                  {isUploading ? 'Uploading Document...' : 'Upload Documents (.txt, .pdf, .docx, .md)'}
                </span>
                <span className="text-[11px] text-slate-500 mt-1">
                  Drag and drop files or click to browse
                </span>
                <input
                  type="file"
                  multiple
                  accept=".txt,.pdf,.docx,.md"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                  className="hidden"
                />
              </label>

              {/* Document List */}
              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {documents.length === 0 ? (
                  <div className="text-center py-6 border border-slate-800/40 rounded-lg text-slate-500 text-xs">
                    No documents uploaded yet. Upload a document to start context retrieval.
                  </div>
                ) : (
                  documents.map((doc) => (
                    <div
                      key={doc.filename}
                      className="flex items-center justify-between p-2.5 rounded-lg bg-[#0e1017] border border-[#1b2030] text-xs"
                    >
                      <div className="flex items-center gap-3 truncate">
                        <FileText className="w-4 h-4 text-blue-400 shrink-0" />
                        <span className="font-mono text-slate-200 truncate">{doc.filename}</span>
                        <span className="px-1.5 py-0.5 rounded bg-[#161a28] border border-[#232a40] text-[10px] font-mono text-slate-400">
                          {doc.file_type}
                        </span>
                        <span className="text-slate-500 text-[11px] font-mono">{doc.size_formatted}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-[11px] font-mono text-slate-400">
                          {doc.chunk_count} chunks
                        </span>
                        <button
                          onClick={() => handleDeleteDocument(doc.filename)}
                          className="text-slate-500 hover:text-rose-400 p-1 transition-colors"
                          title="Delete Document"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Explicit Analyze & Index Button */}
              {documents.length > 0 && (
                <button
                  onClick={handleAnalyzeAndIndex}
                  disabled={isAnalyzingDoc}
                  className="w-full mt-3 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium text-xs transition-colors font-mono shadow-md shadow-emerald-950/30"
                >
                  {isAnalyzingDoc ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-white" />
                      <span>Analyzing Document & Storing Chunks in FAISS...</span>
                    </>
                  ) : (
                    <>
                      <Cpu className="w-4 h-4" />
                      <span>ANALYZE DOCUMENT & BUILD FAISS INDEX</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Configuration Settings (1 col) */}
          <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Sliders className="w-4 h-4 text-blue-400" />
                <h2 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                  Configuration
                </h2>
              </div>

              <div className="space-y-4 text-xs">
                {/* Top K */}
                <div>
                  <div className="flex justify-between text-slate-300 font-mono mb-1">
                    <span>Retrieved Chunks (Top-K):</span>
                    <span className="text-blue-400 font-bold">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full accent-blue-500 bg-[#191d2c] h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Token Budget */}
                <div>
                  <div className="flex justify-between text-slate-300 font-mono mb-1">
                    <span>Token Budget:</span>
                    <span className="text-blue-400 font-bold">{tokenBudget} tokens</span>
                  </div>
                  <input
                    type="range"
                    min="32"
                    max="512"
                    step="16"
                    value={tokenBudget}
                    onChange={(e) => setTokenBudget(parseInt(e.target.value))}
                    className="w-full accent-blue-500 bg-[#191d2c] h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Coverage Threshold */}
                <div>
                  <div className="flex justify-between text-slate-300 font-mono mb-1">
                    <span>Coverage Threshold:</span>
                    <span className="text-blue-400 font-bold">{Math.round(coverageThreshold * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.50"
                    max="1.00"
                    step="0.05"
                    value={coverageThreshold}
                    onChange={(e) => setCoverageThreshold(parseFloat(e.target.value))}
                    className="w-full accent-blue-500 bg-[#191d2c] h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Benchmark Runs */}
                <div>
                  <div className="flex justify-between text-slate-300 font-mono mb-1">
                    <span>Benchmark Runs:</span>
                    <span className="text-blue-400 font-bold">{benchmarkRuns}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={benchmarkRuns}
                    onChange={(e) => setBenchmarkRuns(parseInt(e.target.value))}
                    className="w-full accent-blue-500 bg-[#191d2c] h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-[#1b2030] text-[11px] text-slate-500">
              Evaluates context using Relevance, Evidence, Redundancy, and Token Cost.
            </div>
          </div>
        </div>

        {/* --------------------------------------------------------- */}
        {/* QUERY WORKSPACE */}
        {/* --------------------------------------------------------- */}
        <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Search className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
              Query Workspace
            </h2>
          </div>

          <div>
            <textarea
              rows="3"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Write your Query here"
              className="w-full p-3.5 rounded-lg bg-[#0c0e15] border border-[#202538] focus:border-blue-500 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition-colors font-sans resize-none"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={handleRunAnalysis}
              disabled={isQuerying || isBenchmarking}
              className="flex-1 min-w-[200px] flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-sm transition-colors shadow-lg shadow-blue-900/20"
            >
              {isQuerying ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing TokenWise Pipeline...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>RUN ANALYSIS</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunBenchmark}
              disabled={isQuerying || isBenchmarking}
              className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-[#181c2b] border border-[#2a3148] hover:bg-[#20263a] disabled:opacity-50 text-slate-200 font-medium text-sm transition-colors"
            >
              {isBenchmarking ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-blue-400" />
                  <span>Running LLM Benchmark ({benchmarkRuns} Runs)...</span>
                </>
              ) : (
                <>
                  <BarChart2 className="w-4 h-4 text-blue-400" />
                  <span>RUN BENCHMARK</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* --------------------------------------------------------- */}
        {/* QUERY RESULT DASHBOARD */}
        {/* --------------------------------------------------------- */}
        {queryResult && (
          <div className="space-y-8 animate-fadeIn">
            {/* 1. PRIMARY METRICS BAR */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              {/* Original Tokens */}
              <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-4 flex flex-col justify-between">
                <span className="text-xs text-slate-400 font-medium">Original Tokens</span>
                <span className="text-2xl font-bold font-mono text-slate-100 mt-1">
                  {queryResult.original_tokens}
                </span>
                <span className="text-[11px] text-slate-500 font-mono mt-1">Retrieved Context</span>
              </div>

              {/* Compressed Tokens */}
              <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-4 flex flex-col justify-between">
                <span className="text-xs text-slate-400 font-medium">Compressed Tokens</span>
                <span className="text-2xl font-bold font-mono text-blue-400 mt-1">
                  {queryResult.compressed_tokens}
                </span>
                <span className="text-[11px] text-slate-500 font-mono mt-1">
                  Budget: {queryResult.token_budget}
                </span>
              </div>

              {/* Tokens Saved */}
              <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-4 flex flex-col justify-between">
                <span className="text-xs text-slate-400 font-medium">Tokens Saved</span>
                <span className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                  {queryResult.tokens_saved}
                </span>
                <span className="text-[11px] text-emerald-400/90 font-mono mt-1">
                  -{queryResult.compression_ratio}% tokens
                </span>
              </div>

              {/* Compression % */}
              <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-4 flex flex-col justify-between">
                <span className="text-xs text-slate-400 font-medium">Compression %</span>
                <span className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                  {queryResult.compression_ratio}%
                </span>
                <span className="text-[11px] text-slate-500 font-mono mt-1">Context Reduction</span>
              </div>

              {/* Query Coverage */}
              <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-4 flex flex-col justify-between">
                <span className="text-xs text-slate-400 font-medium">Query Coverage</span>
                <span className="text-2xl font-bold font-mono text-slate-100 mt-1">
                  {Math.round(queryResult.coverage * 100)}%
                </span>
                <span className="text-[11px] text-slate-500 font-mono mt-1">
                  Target: {Math.round(queryResult.coverage_threshold * 100)}%
                </span>
              </div>

              {/* Coverage Guard Status */}
              <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-4 flex flex-col justify-between">
                <span className="text-xs text-slate-400 font-medium">Coverage Guard</span>
                <div className="flex items-center gap-2 mt-1">
                  {queryResult.coverage_guard_passed ? (
                    <span className="inline-flex items-center gap-1.5 text-xs font-bold px-2.5 py-1 rounded bg-emerald-950/60 border border-emerald-800/60 text-emerald-400">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      PASS
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 text-xs font-bold px-2.5 py-1 rounded bg-rose-950/60 border border-rose-800/60 text-rose-400">
                      <XCircle className="w-3.5 h-3.5" />
                      FAIL
                    </span>
                  )}
                </div>
                <span className="text-[11px] text-slate-500 font-mono mt-1">
                  {queryResult.coverage_guard_triggered ? 'Guard Triggered' : 'Normal Selection'}
                </span>
              </div>
            </div>

            {/* 2. OPENROUTER GROUNDED LLM ANSWER */}
            <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-[#1c2132] pb-3">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-blue-400" />
                  <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                    OpenRouter LLM Answer
                  </h3>
                </div>

                {queryResult.llm_answer && (
                  <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
                    <span>Latency: <strong className="text-slate-200">{queryResult.llm_latency_ms} ms</strong></span>
                    <span>•</span>
                    <span>Tokens: <strong className="text-slate-200">{queryResult.llm_total_tokens}</strong> ({queryResult.llm_prompt_tokens} prompt + {queryResult.llm_completion_tokens} completion)</span>
                  </div>
                )}
              </div>

              {queryResult.llm_error ? (
                <div className="p-4 rounded-lg bg-amber-950/30 border border-amber-800/40 text-amber-200 text-xs leading-relaxed space-y-1 font-mono">
                  <div className="font-bold flex items-center gap-2 text-amber-400">
                    <AlertTriangle className="w-4 h-4" />
                    OpenRouter Integration Status:
                  </div>
                  <div>{queryResult.llm_error}</div>
                  <div className="text-[11px] text-slate-400 pt-1">
                    TokenWise context compression executed cleanly above. To generate live LLM responses, add your <code className="text-amber-300">OPENROUTER_API_KEY</code> to the backend <code className="text-amber-300">.env</code> file.
                  </div>
                </div>
              ) : (
                <div className="p-4 rounded-lg bg-[#0c0e15] border border-[#1d2234] text-sm text-slate-100 leading-relaxed font-sans">
                  {queryResult.llm_answer}
                </div>
              )}
            </div>

            {/* 3. CONTEXT COMPARISON */}
            <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-[#1c2132] pb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-blue-400" />
                  <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                    Context Analysis: Original vs TokenWise Compressed
                  </h3>
                </div>
                <button
                  onClick={() => setShowOriginalContext(!showOriginalContext)}
                  className="text-xs px-3 py-1 rounded bg-[#181c2b] border border-[#252b3e] text-slate-300 hover:bg-[#20263a] transition-colors"
                >
                  {showOriginalContext ? 'Hide Full Context' : 'Toggle Raw Expand'}
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Original Context */}
                <div className="flex flex-col space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono text-slate-400 bg-[#0c0e15] px-3 py-2 rounded-t-lg border border-[#1d2234]">
                    <span>Original Retrieved Context</span>
                    <span className="text-slate-300 font-bold">{queryResult.original_tokens} tokens</span>
                  </div>
                  <div className="p-3.5 bg-[#0a0b10] border border-[#1b2030] rounded-b-lg font-mono text-xs text-slate-300 overflow-y-auto max-h-60 leading-relaxed whitespace-pre-wrap">
                    {queryResult.retrieved_context}
                  </div>
                </div>

                {/* Compressed Context */}
                <div className="flex flex-col space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono text-blue-400 bg-[#0c0e15] px-3 py-2 rounded-t-lg border border-[#1d2234]">
                    <span>TokenWise Compressed Context</span>
                    <span className="font-bold">{queryResult.compressed_tokens} tokens</span>
                  </div>
                  <div className="p-3.5 bg-[#0a0b10] border border-blue-900/30 rounded-b-lg font-mono text-xs text-blue-100 overflow-y-auto max-h-60 leading-relaxed whitespace-pre-wrap">
                    {queryResult.compressed_context}
                  </div>
                </div>
              </div>
            </div>

            {/* 4. SENTENCE COMPRESSION DECISIONS */}
            <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-[#1c2132] pb-3">
                <div className="flex items-center gap-2">
                  <FileCheck className="w-4 h-4 text-blue-400" />
                  <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                    Sentence Compression Decisions ({queryResult.total_sentences} Sentences: {queryResult.kept_sentences} Kept, {queryResult.removed_sentences} Removed)
                  </h3>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* KEPT SENTENCES */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-950/20 px-3 py-2 rounded border border-emerald-900/30">
                    <span>Kept Sentences ({queryResult.kept?.length || 0})</span>
                  </div>

                  <div className="space-y-2">
                    {queryResult.kept?.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-[#0c0e15] border border-emerald-900/40 space-y-2 text-xs"
                      >
                        <div className="text-slate-200 font-sans leading-relaxed">{item.sentence}</div>
                        <div className="grid grid-cols-3 gap-2 font-mono text-[11px] pt-2 border-t border-[#181d2c] text-slate-400">
                          <div>Rel: <span className="text-slate-200">{item.relevance?.toFixed(3)}</span></div>
                          <div>Evid: <span className="text-slate-200">{item.evidence?.toFixed(3)}</span></div>
                          <div>Redun: <span className="text-slate-200">{item.redundancy?.toFixed(3)}</span></div>
                          <div>Score: <span className="text-emerald-400 font-bold">{item.score?.toFixed(3)}</span></div>
                          <div>Cost: <span className="text-slate-200">{item.token_cost} tks</span></div>
                          <div>Val: <span className="text-blue-400 font-bold">{item.token_value?.toFixed(5)}</span></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* REMOVED SENTENCES */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-rose-400 bg-rose-950/20 px-3 py-2 rounded border border-rose-900/30">
                    <span>Removed Sentences ({queryResult.removed?.length || 0})</span>
                  </div>

                  <div className="space-y-2">
                    {queryResult.removed?.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-[#0c0e15] border border-rose-900/40 space-y-2 text-xs opacity-75 hover:opacity-100 transition-opacity"
                      >
                        <div className="text-slate-400 font-sans leading-relaxed line-through">{item.sentence}</div>
                        <div className="grid grid-cols-3 gap-2 font-mono text-[11px] pt-2 border-t border-[#181d2c] text-slate-500">
                          <div>Rel: <span>{item.relevance?.toFixed(3)}</span></div>
                          <div>Evid: <span>{item.evidence?.toFixed(3)}</span></div>
                          <div>Redun: <span>{item.redundancy?.toFixed(3)}</span></div>
                          <div>Score: <span className="text-rose-400">{item.score?.toFixed(3)}</span></div>
                          <div>Cost: <span>{item.token_cost} tks</span></div>
                          <div>Val: <span>{item.token_value?.toFixed(5)}</span></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* 5. RETRIEVED CHUNKS */}
            <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-[#1c2132] pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-400" />
                  <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                    FAISS Retrieved Chunks ({queryResult.retrieved_chunk_count})
                  </h3>
                </div>
              </div>

              <div className="space-y-3">
                {queryResult.retrieved_chunks?.map((chunk) => {
                  const isExpanded = expandedChunks[chunk.chunk_id];
                  return (
                    <div
                      key={chunk.chunk_id || chunk.rank}
                      className="border border-[#1d2234] bg-[#0c0e15] rounded-lg p-4 space-y-2 text-xs"
                    >
                      <div className="flex items-center justify-between font-mono">
                        <div className="flex items-center gap-3">
                          <span className="px-2 py-0.5 rounded bg-blue-950/60 border border-blue-800/60 text-blue-400 font-bold">
                            Rank #{chunk.rank}
                          </span>
                          <span className="text-slate-300 font-semibold">{chunk.document_name}</span>
                          <span className="text-slate-500">ID: {chunk.chunk_id}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-slate-400">FAISS Score: <strong className="text-emerald-400">{chunk.score?.toFixed(4)}</strong></span>
                          <button
                            onClick={() =>
                              setExpandedChunks((prev) => ({
                                ...prev,
                                [chunk.chunk_id]: !prev[chunk.chunk_id],
                              }))
                            }
                            className="text-slate-400 hover:text-slate-200"
                          >
                            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          </button>
                        </div>
                      </div>

                      <div className={`font-mono text-slate-300 leading-relaxed pt-2 border-t border-[#181c2b] ${!isExpanded && 'line-clamp-2'}`}>
                        {chunk.text}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 6. PIPELINE PERFORMANCE */}
            <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-2 border-b border-[#1c2132] pb-3">
                <Clock className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
                  Pipeline Latency & Performance Breakdown
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-3.5 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                  <div className="text-slate-400 mb-1">Retrieval Time</div>
                  <div className="text-lg font-bold text-slate-100">{queryResult.retrieval_time_ms} ms</div>
                </div>
                <div className="p-3.5 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                  <div className="text-slate-400 mb-1">Compression Time</div>
                  <div className="text-lg font-bold text-slate-100">{queryResult.compression_time_ms} ms</div>
                </div>
                <div className="p-3.5 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                  <div className="text-slate-400 mb-1">Total Preprocessing</div>
                  <div className="text-lg font-bold text-blue-400">{queryResult.total_preprocessing_time_ms} ms</div>
                </div>
                <div className="p-3.5 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                  <div className="text-slate-400 mb-1">LLM Inference Latency</div>
                  <div className="text-lg font-bold text-emerald-400">{queryResult.llm_latency_ms} ms</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* --------------------------------------------------------- */}
        {/* BENCHMARK DASHBOARD */}
        {/* --------------------------------------------------------- */}
        {benchmarkResult && (
          <div className="bg-[#10121a] border border-blue-900/40 rounded-xl p-6 space-y-6 animate-fadeIn">
            <div className="flex items-center justify-between border-b border-[#1c2132] pb-4">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-blue-400" />
                <h3 className="text-base font-bold text-white uppercase tracking-wide">
                  LLM Comparative Benchmark Output ({benchmarkResult.runs} Runs)
                </h3>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-slate-400">Answer Consistency:</span>
                <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded border ${
                  benchmarkResult.answer_consistency === 'MATCH'
                    ? 'bg-emerald-950/60 border-emerald-800 text-emerald-400'
                    : 'bg-amber-950/60 border-amber-800 text-amber-400'
                }`}>
                  {benchmarkResult.answer_consistency}
                </span>
              </div>
            </div>

            {/* Answer Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans">
              <div className="p-4 rounded-lg bg-[#0c0e15] border border-[#1d2234] space-y-2">
                <div className="font-mono font-semibold text-slate-400 uppercase text-[11px]">Original Context Response</div>
                <div className="text-slate-200 leading-relaxed">{benchmarkResult.original_answer}</div>
              </div>

              <div className="p-4 rounded-lg bg-[#0c0e15] border border-blue-900/40 space-y-2">
                <div className="font-mono font-semibold text-blue-400 uppercase text-[11px]">TokenWise Context Response</div>
                <div className="text-slate-200 leading-relaxed">{benchmarkResult.compressed_answer}</div>
              </div>
            </div>

            {/* Benchmark Latency & Token Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-4 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                <div className="text-slate-400">Original Avg Latency</div>
                <div className="text-xl font-bold text-slate-100 mt-1">{benchmarkResult.original_avg_latency_ms} ms</div>
                <div className="text-[11px] text-slate-500 mt-1">Range: {benchmarkResult.original_min_latency_ms} - {benchmarkResult.original_max_latency_ms} ms</div>
              </div>

              <div className="p-4 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                <div className="text-slate-400">Compressed Avg Latency</div>
                <div className="text-xl font-bold text-blue-400 mt-1">{benchmarkResult.compressed_avg_latency_ms} ms</div>
                <div className="text-[11px] text-slate-500 mt-1">Range: {benchmarkResult.compressed_min_latency_ms} - {benchmarkResult.compressed_max_latency_ms} ms</div>
              </div>

              <div className="p-4 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                <div className="text-slate-400">Prompt Tokens Saved</div>
                <div className="text-xl font-bold text-emerald-400 mt-1">{benchmarkResult.prompt_tokens_saved}</div>
                <div className="text-[11px] text-emerald-400 mt-1">-{benchmarkResult.prompt_token_reduction_pct}% Prompt Tokens</div>
              </div>

              <div className="p-4 rounded-lg bg-[#0c0e15] border border-[#1d2234]">
                <div className="text-slate-400">Latency Assessment</div>
                <div className="text-sm font-bold text-slate-200 mt-2">{benchmarkResult.latency_label}</div>
                <div className="text-[11px] text-slate-500 mt-1">Diff: {benchmarkResult.latency_diff_ms} ms</div>
              </div>
            </div>
          </div>
        )}

        {/* --------------------------------------------------------- */}
        {/* PIPELINE ARCHITECTURE FLOW */}
        {/* --------------------------------------------------------- */}
        <div className="bg-[#10121a] border border-[#1f2434] rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-[#1c2132] pb-3">
            <Info className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-200">
              TokenWise Core Pipeline Architecture
            </h3>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-300 py-2 overflow-x-auto">
            {['Documents', 'Ingestion', 'Chunking', 'FAISS Index', 'Cross-Encoder', 'Evidence Bonus', 'Redundancy', 'Token Optimizer', 'Coverage Guard', 'Compressed Context', 'OpenRouter LLM', 'Answer'].map((step, i, arr) => (
              <React.Fragment key={step}>
                <div className="px-3 py-1.5 rounded-lg bg-[#0c0e15] border border-[#20263a] text-center shrink-0">
                  {step}
                </div>
                {i < arr.length - 1 && <span className="text-slate-600 font-bold shrink-0">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* --------------------------------------------------------- */}
        {/* FINAL EVALUATION BANNER */}
        {/* --------------------------------------------------------- */}
        {queryResult && (
          <div className="bg-[#0c0e15] border border-[#1d2234] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
            <div className="flex items-center gap-6">
              <div>Context Compression: <span className="text-emerald-400 font-bold">{queryResult.compression_ratio}%</span></div>
              <div>Query Coverage: <span className="text-slate-100 font-bold">{Math.round(queryResult.coverage * 100)}%</span></div>
              <div>Coverage Guard: <span className="text-emerald-400 font-bold">{queryResult.coverage_guard_passed ? 'PASS' : 'FAIL'}</span></div>
              <div>Preprocessing Time: <span className="text-blue-400 font-bold">{queryResult.total_preprocessing_time_ms} ms</span></div>
            </div>
            <div className="text-slate-500">
              TokenWise Smart Context Compression Engine
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
