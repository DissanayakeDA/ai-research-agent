"""Streamlit UI: upload PDFs, ask questions, and inspect the evidence behind every answer.

    streamlit run frontend/streamlit_app.py      (start the API first: uvicorn app.main:app)

The UI talks to the FastAPI backend over HTTP only - it never imports the RAG code - so
it stays a thin layer, and the backend could serve other clients too.
"""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
ASK_TIMEOUT = 180  # waiting out an LLM rate limit can take up to a minute
UPLOAD_TIMEOUT = 900  # embedding 20 papers on a CPU takes about a minute


def api(method: str, path: str, **kwargs):
    """Call the backend. On failure, show a readable error in the page and return None."""
    try:
        response = httpx.request(method, API_URL + path, **kwargs)
    except httpx.TimeoutException:
        st.error(f"The API at {API_URL} took too long to respond.")
        return None
    except httpx.HTTPError:
        st.error(f"Cannot reach the API at {API_URL}. Start it in another terminal with: `uvicorn app.main:app`")
        return None
    if response.status_code >= 400:
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError):
            detail = response.text[:300]
        st.error(f"API error {response.status_code}: {detail}")
        return None
    return response.json()


def pages(item: dict) -> str:
    start, end = item["page_start"], item["page_end"]
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


def sidebar(health: dict) -> tuple[int, float | None]:
    """Index information and retrieval settings. Returns (top_k, min_score)."""
    with st.sidebar:
        st.header("Index")
        left, right = st.columns(2)
        left.metric("Documents", health["documents"])
        right.metric("Chunks", health["chunks"])
        st.caption(
            f"Embedding model: `{health['embedding_model']}`  \n"
            f"Chunks: {health['chunk_size']} characters, {health['chunk_overlap']} overlap. "
            "Chunk size is fixed when documents are indexed: change CHUNK_SIZE in .env, "
            "restart the API and index the PDFs again."
        )

        st.header("Retrieval")
        top_k = st.slider("Passages to retrieve (top-k)", 1, 10, min(health["top_k"], 10))
        use_min_score = st.toggle(
            "Require a minimum similarity score",
            help="Passages below the score are dropped. If none remain, the LLM is not called at all.",
        )
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.6, 0.01, disabled=not use_min_score)

        st.header("Language model")
        where = "runs on this machine" if health["llm_is_local"] else "hosted: retrieved passages are sent to the provider"
        st.caption(f"`{health['llm_model']}` ({where})")
    return top_k, (min_score if use_min_score else None)


def ask_tab(top_k: int, min_score: float | None) -> None:
    with st.form("ask"):
        question = st.text_input("Question", placeholder="e.g. Which solver was faster than the genetic algorithm?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        with st.spinner("Retrieving passages and generating an answer..."):
            answer = api("POST", "/ask", json={"question": question, "top_k": top_k, "min_score": min_score}, timeout=ASK_TIMEOUT)
        if answer is not None:
            st.session_state.answer = answer  # survives Streamlit's rerun on every click

    if "answer" in st.session_state:
        show_answer(st.session_state.answer)


def show_answer(answer: dict) -> None:
    st.subheader("Answer")
    if answer["found"]:
        st.markdown(answer["answer"])
    else:
        st.warning(answer["answer"])

    if answer["model"]:
        st.caption(
            f"Retrieval {answer['retrieval_ms']:.0f} ms · {answer['model']} {answer['generation_ms'] / 1000:.1f} s · "
            f"{answer['prompt_tokens']:,} prompt + {answer['completion_tokens']:,} completion tokens"
        )
    else:
        st.caption(f"Retrieval {answer['retrieval_ms']:.0f} ms · no passage was relevant enough, so the LLM was not called")

    if answer["invalid_citations"]:
        st.error(f"The answer cites passage(s) {answer['invalid_citations']} that were never sent to the model.")
    if answer["found"] and not answer["cited"]:
        st.warning("The answer cites no passage, so nothing in it can be traced to your documents.")

    if answer["sources"]:
        st.subheader("Sources" if answer["found"] else "Passages mentioned (none of them answers the question)")
        for source in answer["sources"]:
            st.markdown(f"**[{source['rank']}]** {source['filename']}, {pages(source)} · similarity {source['score']:.3f}")

    if answer["retrieved"]:
        st.subheader("Retrieved passages")
        st.caption("All of these were sent to the language model. Cited ones are what the answer relies on.")
        for chunk in answer["retrieved"]:
            cited = chunk["rank"] in answer["cited"]
            label = f"[{chunk['rank']}] {'CITED' if cited else 'not cited'} · {chunk['filename']} · {pages(chunk)} · score {chunk['score']:.3f}"
            with st.expander(label):
                st.code(chunk["text"], language=None, wrap_lines=True)

    if answer.get("reasoning"):
        with st.expander("Model reasoning (for debugging; not part of the answer)"):
            st.code(answer["reasoning"], language=None, wrap_lines=True)


def documents_tab(health: dict) -> None:
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        help=f"Up to {health['max_upload_mb']} MB per file. Files already in the index are recognised and skipped.",
    )
    if uploads and st.button("Index these files", type="primary"):
        files = [("files", (upload.name, upload.getvalue(), "application/pdf")) for upload in uploads]
        with st.spinner(f"Indexing {len(files)} file(s): extracting text, chunking, embedding..."):
            results = api("POST", "/documents", files=files, timeout=UPLOAD_TIMEOUT)
        if results is not None:
            st.session_state.upload_results = results
            st.rerun()  # refresh the document counts in the sidebar

    results = st.session_state.get("upload_results")
    if results:
        counts = {status: sum(r["status"] == status for r in results) for status in ("indexed", "skipped", "failed")}
        summary = f"{counts['indexed']} indexed, {counts['skipped']} skipped, {counts['failed']} failed"
        (st.error if counts["failed"] else st.success)(summary)
        st.dataframe(
            [
                {"file": r["filename"], "status": r["status"], "details": r["message"],
                 "pages": r["pages"], "chunks": r["chunks"], "seconds": r["seconds"]}
                for r in results
            ],
            hide_index=True,
        )

    st.subheader("Documents in the index")
    documents = api("GET", "/documents", timeout=30)
    if documents:
        st.dataframe(
            [{"file": d["filename"], "chunks": d["chunk_count"], "document id": d["doc_id"][:12]} for d in documents],
            hide_index=True,
        )
    elif documents == []:
        st.info("No documents yet. Upload some PDFs above.")


st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.title("RAG Document Assistant")
st.caption("Answers come only from passages retrieved from your PDFs, with the sources they cite.")

health = api("GET", "/health", timeout=10)
if health is None:
    st.stop()
top_k, min_score = sidebar(health)
ask, documents = st.tabs(["Ask", "Documents"])
with ask:
    ask_tab(top_k, min_score)
with documents:
    documents_tab(health)
