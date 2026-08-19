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
# HELPER FUNCTIONS
# =========================================================

def average(values):
    """Return the average of a list safely."""

    if not values:
        return 0.0

    return sum(values) / len(values)


def run_llm_benchmark(
    llm,
    query,
    original_context,
    compressed_context,
    runs
):
    """
    Run multiple LLM benchmarks for both contexts.

    Returns:
        original_results
        compressed_results
    """

    original_results = []
    compressed_results = []

    # -----------------------------------------------------
    # Original context
    # -----------------------------------------------------

    for run_number in range(runs):

        result = llm.generate(
            query=query,
            context=original_context
        )

        result["run"] = run_number + 1

        original_results.append(
            result
        )

    # -----------------------------------------------------
    # Compressed context
    # -----------------------------------------------------

    for run_number in range(runs):

        result = llm.generate(
            query=query,
            context=compressed_context
        )

        result["run"] = run_number + 1

        compressed_results.append(
            result
        )

    return (
        original_results,
        compressed_results
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
    # RETRIEVAL + COMPRESSION
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

    coverage_passed = result.get(
        "coverage_guard_passed",
        False
    )

    coverage_triggered = result.get(
        "coverage_guard_triggered",
        False
    )

    missing_concepts = result.get(
        "missing_concepts",
        []
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

    if coverage_passed:

        st.success(
            "PASS — query information preserved"
        )

    else:

        st.error(
            "FAIL — important query concepts may be missing"
        )

    if coverage_triggered:

        st.warning(
            "Coverage guard recovered additional information."
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
                        "document_name",
                        chunk.get(
                            "filename",
                            ""
                        )
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
    # LLM BENCHMARK
    # =====================================================

    if run_llm:

        st.divider()

        st.markdown(
            "## ⚡ LLM Latency Benchmark"
        )

        st.write(
            f"Benchmarking Llama 3.2 over "
            f"**{benchmark_runs} runs** per context."
        )

        llm = OllamaClient(
            model="llama3.2:latest"
        )

        # -------------------------------------------------
        # Warm-up
        # -------------------------------------------------

        with st.spinner(
            "Warming up Llama 3.2..."
        ):

            warmup = llm.generate(
                query=query,
                context=compressed_context
            )

        warmup_latency = warmup.get(
            "latency_ms",
            0.0
        )

        st.caption(
            f"Warm-up: {warmup_latency:.2f} ms"
        )

        # -------------------------------------------------
        # Benchmark
        # -------------------------------------------------

        with st.spinner(
            f"Running {benchmark_runs} benchmark runs..."
        ):

            (
                original_results,
                compressed_results
            ) = run_llm_benchmark(
                llm=llm,
                query=query,
                original_context=original_context,
                compressed_context=compressed_context,
                runs=benchmark_runs
            )

        # -------------------------------------------------
        # Extract latency
        # -------------------------------------------------

        original_latencies = [
            item["latency_ms"]
            for item in original_results
        ]

        compressed_latencies = [
            item["latency_ms"]
            for item in compressed_results
        ]

        original_avg_latency = average(
            original_latencies
        )

        compressed_avg_latency = average(
            compressed_latencies
        )

        latency_saved = (
            original_avg_latency
            -
            compressed_avg_latency
        )

        if original_avg_latency > 0:

            latency_reduction = (
                latency_saved
                /
                original_avg_latency
            ) * 100

        else:

            latency_reduction = 0.0

        # -------------------------------------------------
        # Token metrics
        # -------------------------------------------------

        original_prompt_tokens = [
            item.get(
                "prompt_tokens",
                0
            )
            for item in original_results
        ]

        compressed_prompt_tokens = [
            item.get(
                "prompt_tokens",
                0
            )
            for item in compressed_results
        ]

        original_completion_tokens = [
            item.get(
                "completion_tokens",
                0
            )
            for item in original_results
        ]

        compressed_completion_tokens = [
            item.get(
                "completion_tokens",
                0
            )
            for item in compressed_results
        ]

        original_total_tokens = [
            item.get(
                "total_tokens",
                0
            )
            for item in original_results
        ]

        compressed_total_tokens = [
            item.get(
                "total_tokens",
                0
            )
            for item in compressed_results
        ]

        original_avg_prompt = average(
            original_prompt_tokens
        )

        compressed_avg_prompt = average(
            compressed_prompt_tokens
        )

        original_avg_completion = average(
            original_completion_tokens
        )

        compressed_avg_completion = average(
            compressed_completion_tokens
        )

        original_avg_total = average(
            original_total_tokens
        )

        compressed_avg_total = average(
            compressed_total_tokens
        )

        prompt_tokens_saved = (
            original_avg_prompt
            -
            compressed_avg_prompt
        )

        prompt_reduction = (
            (
                prompt_tokens_saved
                /
                original_avg_prompt
            ) * 100
            if original_avg_prompt > 0
            else 0.0
        )

        total_tokens_saved = (
            original_avg_total
            -
            compressed_avg_total
        )

        total_reduction = (
            (
                total_tokens_saved
                /
                original_avg_total
            ) * 100
            if original_avg_total > 0
            else 0.0
        )

        # -------------------------------------------------
        # Internal Ollama timing
        # -------------------------------------------------

        original_prompt_eval = [
            item.get(
                "prompt_eval_duration_ms",
                0.0
            )
            for item in original_results
        ]

        compressed_prompt_eval = [
            item.get(
                "prompt_eval_duration_ms",
                0.0
            )
            for item in compressed_results
        ]

        original_generation = [
            item.get(
                "generation_duration_ms",
                0.0
            )
            for item in original_results
        ]

        compressed_generation = [
            item.get(
                "generation_duration_ms",
                0.0
            )
            for item in compressed_results
        ]

        original_prompt_eval_avg = average(
            original_prompt_eval
        )

        compressed_prompt_eval_avg = average(
            compressed_prompt_eval
        )

        original_generation_avg = average(
            original_generation
        )

        compressed_generation_avg = average(
            compressed_generation
        )

        prompt_eval_saved = (
            original_prompt_eval_avg
            -
            compressed_prompt_eval_avg
        )

        if original_prompt_eval_avg > 0:

            prompt_eval_reduction = (
                prompt_eval_saved
                /
                original_prompt_eval_avg
            ) * 100

        else:

            prompt_eval_reduction = 0.0

        # -------------------------------------------------
        # Average latency
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
                f"{original_avg_latency:.0f} ms"
            )

        with latency_col2:

            st.metric(
                "Compressed Average",
                f"{compressed_avg_latency:.0f} ms"
            )

        with latency_col3:

            st.metric(
                "Latency Reduction",
                f"{latency_reduction:.1f}%",
                delta=f"{latency_saved:.0f} ms"
            )

        # -------------------------------------------------
        # Latency range
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
                f"**{min(original_latencies):.2f} ms** "
                f"– "
                f"**{max(original_latencies):.2f} ms**"
            )

        with range_col2:

            st.write(
                f"Compressed: "
                f"**{min(compressed_latencies):.2f} ms** "
                f"– "
                f"**{max(compressed_latencies):.2f} ms**"
            )

        # -------------------------------------------------
        # Latency chart
        # -------------------------------------------------

        latency_df = pd.DataFrame(
            {
                "Run": list(
                    range(
                        1,
                        benchmark_runs + 1
                    )
                ),
                "Original": original_latencies,
                "TokenWise": compressed_latencies
            }
        )

        st.markdown(
            "### Run-by-Run Latency"
        )

        st.line_chart(
            latency_df.set_index("Run")
        )

        # -------------------------------------------------
        # Internal timing
        # -------------------------------------------------

        st.markdown(
            "### 🔬 Ollama Internal Timing"
        )

        timing_col1, timing_col2 = (
            st.columns(2)
        )

        with timing_col1:

            st.metric(
                "Prompt Evaluation Reduction",
                f"{prompt_eval_reduction:.2f}%",
                delta=f"{prompt_eval_saved:.2f} ms"
            )

            st.write(
                f"Original prompt evaluation: "
                f"**{original_prompt_eval_avg:.2f} ms**"
            )

            st.write(
                f"Compressed prompt evaluation: "
                f"**{compressed_prompt_eval_avg:.2f} ms**"
            )

        with timing_col2:

            generation_saved = (
                original_generation_avg
                -
                compressed_generation_avg
            )

            st.metric(
                "Generation Time Difference",
                f"{generation_saved:.2f} ms"
            )

            st.write(
                f"Original generation: "
                f"**{original_generation_avg:.2f} ms**"
            )

            st.write(
                f"Compressed generation: "
                f"**{compressed_generation_avg:.2f} ms**"
            )

        # -------------------------------------------------
        # Token comparison
        # -------------------------------------------------

        st.markdown(
            "## 🪙 LLM Token Usage"
        )

        token_col1, token_col2, token_col3 = (
            st.columns(3)
        )

        with token_col1:

            st.metric(
                "Prompt Tokens Saved",
                f"{prompt_tokens_saved:.0f}",
                delta=f"{prompt_reduction:.2f}%"
            )

        with token_col2:

            st.metric(
                "Total Tokens Saved",
                f"{total_tokens_saved:.0f}",
                delta=f"{total_reduction:.2f}%"
            )

        with token_col3:

            st.metric(
                "Original → Compressed",
                f"{original_avg_total:.0f} → "
                f"{compressed_avg_total:.0f}"
            )

        st.write(
            f"Original average prompt tokens: "
            f"**{original_avg_prompt:.0f}**"
        )

        st.write(
            f"Compressed average prompt tokens: "
            f"**{compressed_avg_prompt:.0f}**"
        )

        st.write(
            f"Original average completion tokens: "
            f"**{original_avg_completion:.0f}**"
        )

        st.write(
            f"Compressed average completion tokens: "
            f"**{compressed_avg_completion:.0f}**"
        )

        # -------------------------------------------------
        # Benchmark table
        # -------------------------------------------------

        st.markdown(
            "### Individual Benchmark Runs"
        )

        benchmark_df = pd.DataFrame(
            {
                "Run": list(
                    range(
                        1,
                        benchmark_runs + 1
                    )
                ),
                "Original Latency (ms)": [
                    round(
                        value,
                        2
                    )
                    for value in original_latencies
                ],
                "Compressed Latency (ms)": [
                    round(
                        value,
                        2
                    )
                    for value in compressed_latencies
                ],
                "Original Prompt Tokens": original_prompt_tokens,
                "Compressed Prompt Tokens": compressed_prompt_tokens,
                "Original Total Tokens": original_total_tokens,
                "Compressed Total Tokens": compressed_total_tokens
            }
        )

        st.dataframe(
            benchmark_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # Answers
        # -------------------------------------------------

        st.markdown(
            "## 🤖 LLM Answers"
        )

        original_answer = (
            original_results[0].get(
                "response",
                ""
            )
            if original_results
            else ""
        )

        compressed_answer = (
            compressed_results[0].get(
                "response",
                ""
            )
            if compressed_results
            else ""
        )

        answer_col1, answer_col2 = (
            st.columns(2)
        )

        with answer_col1:

            st.markdown(
                "### Original Context"
            )

            st.info(
                original_answer
            )

        with answer_col2:

            st.markdown(
                "### TokenWise Context"
            )

            st.success(
                compressed_answer
            )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        st.markdown(
            "### Validation"
        )

        if compressed_tokens < original_tokens:

            st.success(
                "PASS: TokenWise reduced context size."
            )

        else:

            st.error(
                "FAIL: Context was not reduced."
            )

        if compressed_avg_prompt < original_avg_prompt:

            st.success(
                "PASS: LLM prompt tokens reduced."
            )

        else:

            st.error(
                "FAIL: LLM prompt tokens were not reduced."
            )

        if coverage_passed:

            st.success(
                "PASS: Query coverage preserved."
            )

        else:

            st.warning(
                "WARNING: Query coverage threshold not satisfied."
            )

        if compressed_avg_latency < original_avg_latency:

            st.success(
                "PASS: Average LLM latency decreased."
            )

        else:

            st.info(
                "INFO: Average LLM latency did not decrease "
                "in this benchmark."
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