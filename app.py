import streamlit as st
from rag_engine import (
    add_document,
    get_documents_list,
    delete_document,
    semantic_search,
    generate_answer,
    collection
)
from evaluator import run_full_evaluation
from test_questions import DEFAULT_QUESTIONS

st.set_page_config(page_title="RAG Evaluator", page_icon="📊")
st.title("📊 RAG Evaluation System")
st.caption("Measure faithfulness, relevance, precision and recall")

# sidebar
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"]
    )
    if uploaded_file:
        if st.button("➕ Add to Knowledge Base"):
            with st.spinner("Processing..."):
                msg = add_document(uploaded_file)
            st.write(msg)
            st.rerun()

    st.divider()

    st.header("📄 Documents")
    docs = get_documents_list()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {doc}")
            with col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    delete_document(doc)
                    st.rerun()
    else:
        st.info("No documents yet.")

    st.divider()
    st.metric("Total Chunks", collection.count())
    st.caption("Day 15 — 30 Day AI Bootcamp")

# tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Q&A", "📊 Evaluate", "📋 Batch Eval"
])

# tab 1 — regular Q&A
with tab1:
    st.subheader("Ask a Question")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask anything...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({
            "role": "user", "content": question
        })
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results = semantic_search(question)
                answer, context = generate_answer(question, results)
            st.write(answer)
        st.session_state.chat_history.append({
            "role": "assistant", "content": answer
        })

# tab 2 — single question evaluation
with tab2:
    st.subheader("Evaluate Single Question")
    st.caption("Ask a question and get quality scores")

    eval_question = st.text_input(
        "Enter question to evaluate:",
        placeholder="What are the main skills mentioned?"
    )

    if st.button("🔍 Run Evaluation", key="single_eval"):
        if not eval_question.strip():
            st.warning("Please enter a question.")
        elif collection.count() == 0:
            st.warning("Please upload a document first.")
        else:
            with st.spinner("Running evaluation — this takes ~15 seconds..."):
                results = semantic_search(eval_question)
                answer, context = generate_answer(
                    eval_question, results
                )
                eval_results = run_full_evaluation(
                    eval_question, context, answer
                )

            # overall score
            overall = eval_results["overall_score"]
            color = (
                "🟢" if overall >= 0.7
                else "🟡" if overall >= 0.4
                else "🔴"
            )
            st.markdown(f"## {color} Overall Score: {overall}/1.0")
            st.divider()

            # answer
            st.markdown("**Generated Answer:**")
            st.info(answer)
            st.divider()

            # metric cards
            st.markdown("**📊 Metric Breakdown:**")
            metrics = eval_results["metrics"]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                score = metrics["faithfulness"]["score"]
                st.metric("Faithfulness", f"{score}/1.0")
                st.caption(metrics["faithfulness"]["reason"])

            with col2:
                score = metrics["answer_relevance"]["score"]
                st.metric("Answer Relevance", f"{score}/1.0")
                st.caption(metrics["answer_relevance"]["reason"])

            with col3:
                score = metrics["context_precision"]["score"]
                st.metric("Context Precision", f"{score}/1.0")
                st.caption(metrics["context_precision"]["reason"])

            with col4:
                score = metrics["context_recall"]["score"]
                st.metric("Context Recall", f"{score}/1.0")
                st.caption(metrics["context_recall"]["reason"])

# tab 3 — batch evaluation
with tab3:
    st.subheader("Batch Evaluation")
    st.caption("Run multiple questions and get average scores")

    # editable test questions
    st.markdown("**Test Questions:**")
    questions_text = st.text_area(
        "One question per line:",
        value="\n".join(DEFAULT_QUESTIONS),
        height=150
    )

    if st.button("🚀 Run Batch Evaluation", key="batch_eval"):
        if collection.count() == 0:
            st.warning("Please upload a document first.")
        else:
            questions = [
                q.strip() for q in questions_text.split("\n")
                if q.strip()
            ]
            all_results = []
            progress = st.progress(0)
            status = st.empty()

            for i, q in enumerate(questions):
                status.text(f"Evaluating: {q[:50]}...")
                results = semantic_search(q)
                answer, context = generate_answer(q, results)
                eval_result = run_full_evaluation(q, context, answer)
                all_results.append(eval_result)
                progress.progress((i + 1) / len(questions))

            status.text("✅ Evaluation complete!")

            # average scores
            avg_overall = round(
                sum(r["overall_score"] for r in all_results)
                / len(all_results), 2
            )
            avg_faith = round(
                sum(r["metrics"]["faithfulness"]["score"]
                    for r in all_results) / len(all_results), 2
            )
            avg_rel = round(
                sum(r["metrics"]["answer_relevance"]["score"]
                    for r in all_results) / len(all_results), 2
            )
            avg_prec = round(
                sum(r["metrics"]["context_precision"]["score"]
                    for r in all_results) / len(all_results), 2
            )
            avg_rec = round(
                sum(r["metrics"]["context_recall"]["score"]
                    for r in all_results) / len(all_results), 2
            )

            st.divider()
            st.markdown("### 📊 Batch Results Summary")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Overall", f"{avg_overall}/1.0")
            with col2:
                st.metric("Faithfulness", f"{avg_faith}/1.0")
            with col3:
                st.metric("Relevance", f"{avg_rel}/1.0")
            with col4:
                st.metric("Precision", f"{avg_prec}/1.0")
            with col5:
                st.metric("Recall", f"{avg_rec}/1.0")

            st.divider()

            # per question results
            st.markdown("### 📋 Per Question Results")
            for r in all_results:
                overall = r["overall_score"]
                color = (
                    "🟢" if overall >= 0.7
                    else "🟡" if overall >= 0.4
                    else "🔴"
                )
                with st.expander(
                    f"{color} {r['question'][:60]}... "
                    f"— {overall}/1.0"
                ):
                    st.write(f"**Answer:** {r['answer']}")
                    st.divider()
                    m = r["metrics"]
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "Faithfulness",
                            m["faithfulness"]["score"]
                        )
                        st.caption(m["faithfulness"]["reason"])
                    with c2:
                        st.metric(
                            "Relevance",
                            m["answer_relevance"]["score"]
                        )
                        st.caption(m["answer_relevance"]["reason"])
                    with c3:
                        st.metric(
                            "Precision",
                            m["context_precision"]["score"]
                        )
                        st.caption(m["context_precision"]["reason"])
                    with c4:
                        st.metric(
                            "Recall",
                            m["context_recall"]["score"]
                        )
                        st.caption(m["context_recall"]["reason"])