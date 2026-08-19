import os
import sys
import time

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# TOKENWISE IMPORTS
# =========================================================

from rag.rag_pipeline import RAGPipeline
from llm.ollama_client import OllamaClient


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TokenWise",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("⚡ TokenWise")

st.subheader(
    "Smart Context Compression for Faster RAG"
)

st.markdown(
    """
TokenWise removes low-value and redundant information from
retrieved RAG context before it reaches the LLM.

**Goal:** fewer tokens → lower latency → lower inference cost.
"""
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Configuration")

documents_dir = st.sidebar.text_input(
    "Documents directory",
    value="documents"
)

top_k = st.sidebar.slider(
    "Retrieved chunks",
    min_value=1,
    max_value=10,
    value=4
)

token_budget = st.sidebar.slider(
    "Token budget",
    min_value=32,
    max_value=512,
    value=80,
    step=16
)

coverage_threshold = st.sidebar.slider(
    "Coverage threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.80,
    step=0.05
)

run_llm = st.sidebar.checkbox(
    "Run Llama 3.2 benchmark",
    value=True
)

benchmark_runs = st.sidebar.slider(
    "LLM benchmark runs",
    min_value=1,
    max_value=10,
    value=5
)


# =========================================================
# QUERY
# =========================================================

st.markdown("### 🔎 Query")

query = st.text_area(
    "Ask a question",
    value=(
        "What caused the increase in renewable energy "
        "production between 2020 and 2025?"
    ),
    height=100
)


# =========================================================
# RUN BUTTON
# =========================================================

run_button = st.button(
    "🚀 Run TokenWise",
    type="primary",
    use_container_width=True
)


# =========================================================
# MAIN PIPELINE
# =========================================================

if run_button:

    if not query.strip():

        st.error(
            "Please enter a query."
        )

        st.stop()

    # =====================================================
    # INITIALIZE TOKENWISE
    # =====================================================

    with st.spinner(
        "Initializing TokenWise..."
    ):

        pipeline = RAGPipeline(
            documents_dir=documents_dir,
            chunk_size=3,
            chunk_overlap=1,
            top_k=top_k,
            token_budget=token_budget,
            coverage_threshold=coverage_threshold
        )

        pipeline.build_index()

    # =====================================================
    # RUN RETRIEVAL + COMPRESSION
    # =====================================================

    with st.spinner(
        "Retrieving and compressing context..."
    ):

        pipeline_start = time.perf_counter()

        result = pipeline.run(
            query=query,
            top_k=top_k,
            token_budget=token_budget
        )

        pipeline_latency = (
            time.perf_counter()
            -
            pipeline_start
        ) * 1000

    # =====================================================
    # BASIC METRICS
    # =====================================================

    original_tokens = result.get(
        "original_tokens",
        0
    )

    compressed_tokens = result.get(
        "compressed_tokens",
        0
    )

    tokens_saved = result.get(
        "tokens_saved",
        original_tokens - compressed_tokens
    )

    compression_ratio = result.get(
        "compression_ratio",
        0.0
    )

    coverage = result.get(
        "coverage",
        0.0
    )

    # =====================================================
    # TOKENWISE METRICS
    # =====================================================

    st.markdown(
        "## 📊 TokenWise Metrics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Original Tokens",
            original_tokens
        )

    with col2:

        st.metric(
            "Compressed Tokens",
            compressed_tokens
        )

    with col3:

        st.metric(
            "Tokens Saved",
            tokens_saved,
            delta=f"{compression_ratio:.1f}%"
        )

    with col4:

        st.metric(
            "Query Coverage",
            f"{coverage:.1%}"
        )

    # =====================================================
    # COMPRESSION
    # =====================================================

    st.markdown(
        "### 🗜️ Context Compression"
    )

    compression_progress = min(
        max(
            compression_ratio / 100,
            0.0
        ),
        1.0
    )

    st.progress(
        compression_progress
    )

    st.write(
        f"**{compression_ratio:.2f}% of context tokens removed**"
    )

    st.write(
        f"Original: **{original_tokens} tokens** "
        f"→ Compressed: **{compressed_tokens} tokens**"
    )

    # =====================================================
    # COVERAGE GUARD
    # =====================================================

    st.markdown(
        "### 🛡️ Coverage Guard"
    )

    coverage_passed = result.get(
        "coverage_guard_passed",
        False
    )

    if coverage_passed:

        st.success(
            "PASS — query information preserved"
        )

    else:

        st.error(
            "FAIL — important query concepts may be missing"
        )

    if result.get(
        "coverage_guard_triggered",
        False
    ):

        st.warning(
            "Coverage guard recovered additional information."
        )

    missing_concepts = result.get(
        "missing_concepts",
        []
    )

    if missing_concepts:

        st.write(
            "Missing concepts:",
            ", ".join(
                missing_concepts
            )
        )

    else:

        st.write(
            "Missing concepts: None"
        )

    # =====================================================
    # RETRIEVED CHUNKS
    # =====================================================

    st.markdown(
        "## 📚 Retrieved Chunks"
    )

    retrieved = (
        result.get("retrieved")
        or result.get("retrieved_chunks")
        or result.get("chunks")
        or []
    )

    if retrieved:

        st.write(
            f"Retrieved **{len(retrieved)} chunks**."
        )

        for index, chunk in enumerate(
            retrieved,
            start=1
        ):

            if isinstance(
                chunk,
                dict
            ):

                score = chunk.get(
                    "score",
                    chunk.get(
                        "similarity",
                        0.0
                    )
                )

                text = chunk.get(
                    "text",
                    chunk.get(
                        "content",
                        chunk.get(
                            "page_content",
                            ""
                        )
                    )
                )

                document = chunk.get(
                    "document",
                    chunk.get(
                        "filename",
                        ""
                    )
                )

                chunk_id = chunk.get(
                    "chunk_id",
                    ""
                )

            else:

                score = 0.0
                text = str(chunk)
                document = ""
                chunk_id = ""

            title = (
                f"Rank {index} "
                f"• Score {float(score):.4f}"
            )

            with st.expander(
                title
            ):

                st.write(
                    text
                )

                metadata = []

                if document:

                    metadata.append(
                        f"Document: {document}"
                    )

                if chunk_id:

                    metadata.append(
                        f"Chunk ID: {chunk_id}"
                    )

                if metadata:

                    st.caption(
                        " | ".join(metadata)
                    )

    else:

        st.info(
            "No retrieved chunk metadata was returned."
        )

    # =====================================================
    # CONTEXT ANALYSIS
    # =====================================================

    st.markdown(
        "## 🧠 Context Analysis"
    )

    original_context = result.get(
        "retrieved_context",
        ""
    )

    compressed_context = result.get(
        "compressed_context",
        ""
    )

    context_col1, context_col2 = (
        st.columns(2)
    )

    with context_col1:

        st.markdown(
            "### Original Retrieved Context"
        )

        st.code(
            original_context,
            language="text"
        )

        st.caption(
            f"{original_tokens} tokens"
        )

    with context_col2:

        st.markdown(
            "### TokenWise Compressed Context"
        )

        st.code(
            compressed_context,
            language="text"
        )

        st.caption(
            f"{compressed_tokens} tokens"
        )

    # =====================================================
    # COMPRESSION DECISIONS
    # =====================================================

    st.markdown(
        "## 🔬 Compression Decisions"
    )

    kept = result.get(
        "kept",
        []
    )

    removed = result.get(
        "removed",
        []
    )

    decision_col1, decision_col2 = (
        st.columns(2)
    )

    # -----------------------------------------------------
    # KEPT
    # -----------------------------------------------------

    with decision_col1:

        st.markdown(
            f"### ✅ Kept ({len(kept)})"
        )

        for candidate in kept:

            sentence = candidate.get(
                "sentence",
                ""
            )

            with st.expander(
                sentence
            ):

                st.write(
                    f"**Relevance:** "
                    f"{candidate.get('relevance', 0.0):.4f}"
                )

                st.write(
                    f"**Evidence:** "
                    f"{candidate.get('evidence', 0.0):.4f}"
                )

                st.write(
                    f"**Redundancy:** "
                    f"{candidate.get('redundancy', 0.0):.4f}"
                )

                st.write(
                    f"**Final Score:** "
                    f"{candidate.get('score', 0.0):.4f}"
                )

                st.write(
                    f"**Tokens:** "
                    f"{candidate.get('token_cost', 0)}"
                )

                st.write(
                    f"**Token Value:** "
                    f"{candidate.get('token_value', 0.0):.6f}"
                )

    # -----------------------------------------------------
    # REMOVED
    # -----------------------------------------------------

    with decision_col2:

        st.markdown(
            f"### ❌ Removed ({len(removed)})"
        )

        for candidate in removed:

            sentence = candidate.get(
                "sentence",
                ""
            )

            with st.expander(
                sentence
            ):

                st.write(
                    f"**Relevance:** "
                    f"{candidate.get('relevance', 0.0):.4f}"
                )

                st.write(
                    f"**Evidence:** "
                    f"{candidate.get('evidence', 0.0):.4f}"
                )

                st.write(
                    f"**Redundancy:** "
                    f"{candidate.get('redundancy', 0.0):.4f}"
                )

                st.write(
                    f"**Final Score:** "
                    f"{candidate.get('score', 0.0):.4f}"
                )

                st.write(
                    f"**Tokens:** "
                    f"{candidate.get('token_cost', 0)}"
                )

                st.write(
                    f"**Token Value:** "
                    f"{candidate.get('token_value', 0.0):.6f}"
                )

    # =====================================================
    # LLM BENCHMARK
    # =====================================================

    if run_llm:

        st.markdown(
            "## ⚡ LLM Latency Benchmark"
        )

        st.caption(
            f"Benchmarking Llama 3.2 over "
            f"**{benchmark_runs} runs** per context."
        )

        llm = OllamaClient(
            model="llama3.2:latest"
        )

        # -------------------------------------------------
        # WARM-UP
        # -------------------------------------------------

        with st.spinner(
            "Warming up Llama 3.2..."
        ):

            warmup = llm.generate(
                query=query,
                context=compressed_context
            )

        st.caption(
            f"Warm-up: "
            f"{warmup['latency_ms']:.2f} ms"
        )

        # -------------------------------------------------
        # ORIGINAL BENCHMARK
        # -------------------------------------------------

        original_latencies = []
        original_results = []

        with st.spinner(
            "Benchmarking original context..."
        ):

            for _ in range(
                benchmark_runs
            ):

                response = llm.generate(
                    query=query,
                    context=original_context
                )

                original_latencies.append(
                    response["latency_ms"]
                )

                original_results.append(
                    response
                )

        # -------------------------------------------------
        # COMPRESSED BENCHMARK
        # -------------------------------------------------

        compressed_latencies = []
        compressed_results = []

        with st.spinner(
            "Benchmarking compressed context..."
        ):

            for _ in range(
                benchmark_runs
            ):

                response = llm.generate(
                    query=query,
                    context=compressed_context
                )

                compressed_latencies.append(
                    response["latency_ms"]
                )

                compressed_results.append(
                    response
                )

        # -------------------------------------------------
        # AVERAGES
        # -------------------------------------------------

        original_average = (
            sum(original_latencies)
            /
            len(original_latencies)
        )

        compressed_average = (
            sum(compressed_latencies)
            /
            len(compressed_latencies)
        )

        original_min = min(
            original_latencies
        )

        original_max = max(
            original_latencies
        )

        compressed_min = min(
            compressed_latencies
        )

        compressed_max = max(
            compressed_latencies
        )

        latency_saved = (
            original_average
            -
            compressed_average
        )

        if original_average > 0:

            latency_reduction = (
                latency_saved
                /
                original_average
            ) * 100

        else:

            latency_reduction = 0.0

        # -------------------------------------------------
        # LATENCY METRICS
        # -------------------------------------------------

        st.markdown(
            "### Average Latency"
        )

        latency_col1, latency_col2, latency_col3 = (
            st.columns(3)
        )

        with latency_col1:

            st.metric(
                "Original Average",
                f"{original_average:.0f} ms"
            )

        with latency_col2:

            st.metric(
                "Compressed Average",
                f"{compressed_average:.0f} ms"
            )

        with latency_col3:

            st.metric(
                "Latency Reduction",
                f"{latency_reduction:.1f}%",
                delta=f"{latency_saved:.0f} ms"
            )

        # -------------------------------------------------
        # MIN / MAX
        # -------------------------------------------------

        st.markdown(
            "### Latency Range"
        )

        range_col1, range_col2 = (
            st.columns(2)
        )

        with range_col1:

            st.write(
                f"Original: "
                f"**{original_min:.0f} ms** "
                f"– "
                f"**{original_max:.0f} ms**"
            )

        with range_col2:

            st.write(
                f"Compressed: "
                f"**{compressed_min:.0f} ms** "
                f"– "
                f"**{compressed_max:.0f} ms**"
            )

        # -------------------------------------------------
        # CLEAN BAR CHART
        # -------------------------------------------------

        latency_df = pd.DataFrame(
            {
                "Context": [
                    "Original",
                    "Compressed"
                ],
                "Average Latency (ms)": [
                    original_average,
                    compressed_average
                ]
            }
        ).set_index(
            "Context"
        )

        st.bar_chart(
            latency_df,
            y="Average Latency (ms)"
        )

        # -------------------------------------------------
        # INDIVIDUAL RUNS
        # -------------------------------------------------

        with st.expander(
            "View individual benchmark runs"
        ):

            benchmark_df = pd.DataFrame(
                {
                    "Run": list(
                        range(
                            1,
                            benchmark_runs + 1
                        )
                    ),
                    "Original (ms)": (
                        original_latencies
                    ),
                    "Compressed (ms)": (
                        compressed_latencies
                    )
                }
            )

            st.dataframe(
                benchmark_df,
                use_container_width=True,
                hide_index=True
            )

        # =================================================
        # TOKEN USAGE
        # =================================================

        st.markdown(
            "### 🪙 LLM Token Usage"
        )

        original_response = (
            original_results[-1]
        )

        compressed_response = (
            compressed_results[-1]
        )

        original_prompt_tokens = (
            original_response.get(
                "prompt_tokens",
                0
            )
        )

        compressed_prompt_tokens = (
            compressed_response.get(
                "prompt_tokens",
                0
            )
        )

        original_completion_tokens = (
            original_response.get(
                "completion_tokens",
                0
            )
        )

        compressed_completion_tokens = (
            compressed_response.get(
                "completion_tokens",
                0
            )
        )

        original_total_tokens = (
            original_response.get(
                "total_tokens",
                original_prompt_tokens
                +
                original_completion_tokens
            )
        )

        compressed_total_tokens = (
            compressed_response.get(
                "total_tokens",
                compressed_prompt_tokens
                +
                compressed_completion_tokens
            )
        )

        prompt_tokens_saved = (
            original_prompt_tokens
            -
            compressed_prompt_tokens
        )

        total_tokens_saved = (
            original_total_tokens
            -
            compressed_total_tokens
        )

        token_col1, token_col2, token_col3 = (
            st.columns(3)
        )

        with token_col1:

            st.metric(
                "Original Prompt",
                original_prompt_tokens
            )

        with token_col2:

            st.metric(
                "Compressed Prompt",
                compressed_prompt_tokens
            )

        with token_col3:

            st.metric(
                "Prompt Tokens Saved",
                prompt_tokens_saved
            )

        st.write(
            f"Original total LLM tokens: "
            f"**{original_total_tokens}**"
        )

        st.write(
            f"Compressed total LLM tokens: "
            f"**{compressed_total_tokens}**"
        )

        st.write(
            f"Total LLM tokens saved: "
            f"**{total_tokens_saved}**"
        )

        # =================================================
        # ANSWERS
        # =================================================

        st.markdown(
            "## 🤖 LLM Answers"
        )

        answer_col1, answer_col2 = (
            st.columns(2)
        )

        with answer_col1:

            st.markdown(
                "### Original Context"
            )

            st.info(
                original_response.get(
                    "response",
                    ""
                )
            )

        with answer_col2:

            st.markdown(
                "### TokenWise Context"
            )

            st.success(
                compressed_response.get(
                    "response",
                    ""
                )
            )

    else:

        st.info(
            "LLM benchmark disabled."
        )

    # =====================================================
    # PIPELINE PERFORMANCE
    # =====================================================

    st.markdown(
        "## ⏱️ Pipeline Performance"
    )

    st.metric(
        "Retrieval + Compression",
        f"{pipeline_latency:.2f} ms"
    )

    st.caption(
        "TokenWise preprocessing only. "
        "LLM latency is benchmarked separately."
    )

    # =====================================================
    # PIPELINE ARCHITECTURE
    # =====================================================

    st.divider()

    st.markdown(
        "## ⚡ TokenWise Pipeline"
    )

    st.markdown(
        """
**Documents**
→ **Chunking**
→ **FAISS Retrieval**
→ **Cross-Encoder Ranking**
→ **Evidence Scoring**
→ **Redundancy Detection**
→ **Token Optimization**
→ **Coverage Guard**
→ **LLM**
"""
    )

    st.markdown(
        """
**Core objective:** send less text to the LLM without
losing the information needed to answer the query.
"""
    )

    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.caption(
        "TokenWise — Smart Context Compression"
    )