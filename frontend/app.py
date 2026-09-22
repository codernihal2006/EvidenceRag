import requests
import streamlit as st

API = st.sidebar.text_input("Backend URL", "http://localhost:8000")
st.set_page_config(page_title="EvidenceRAG", page_icon="📚", layout="wide")
st.title("EvidenceRAG")
st.caption("Hybrid retrieval with grounded AI answers")

with st.sidebar:
    st.header("Documents")
    upload = st.file_uploader("Upload PDF", type="pdf")
    if upload and st.button("Index PDF"):
        with st.spinner("Parsing, chunking, and indexing…"):
            response = requests.post(f"{API}/documents/upload", files={"file": (upload.name, upload.getvalue(), "application/pdf")})
        if response.ok: st.success(f"Indexed {upload.name}")
        else: st.error(response.text)
    st.header("Retrieval settings")
    dense_k = st.number_input("Dense Top-K", 1, 50, 15)
    bm25_k = st.number_input("BM25 Top-K", 1, 50, 15)
    rrf_k = st.number_input("RRF K", 1, 200, 60)
    rerank_k = st.number_input("Reranker Top-K", 1, 10, 5)
    st.divider()
    try:
        docs = requests.get(f"{API}/documents", timeout=3).json()
        st.subheader("Indexed documents")
        for doc in docs:
            st.write(f"**{doc['document_name']}**  ")
            st.caption(f"{doc['pages']} pages · {doc['chunks']} chunks · {doc['status']}")
            if st.button("Delete", key=doc["document_id"]):
                requests.delete(f"{API}/documents/{doc['document_id']}"); st.rerun()
    except requests.RequestException:
        st.warning("Start the backend to manage documents.")

question = st.text_input("Ask a question about your documents…")
if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving evidence and validating citations…"):
        response = requests.post(f"{API}/query", json={"question": question, "dense_top_k": dense_k, "bm25_top_k": bm25_k, "rrf_k": rrf_k, "reranker_top_k": rerank_k})
    if not response.ok:
        st.error(response.text)
    else:
        result = response.json()
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption(" · ".join(f"{key}: {value:.0f} ms" for key, value in result["timings_ms"].items()))
        st.subheader("Sources")
        for source in result["sources"]:
            with st.expander(f"[{source['citation_id']}] {source['chunk']['document_name']} — Page {source['chunk']['page_number']}"):
                st.caption(source["chunk"].get("section", "")); st.write(source["chunk"]["text"])
        with st.expander("Retrieval debugging"):
            for label, key in [("Dense Retrieval", "dense_results"), ("BM25 Retrieval", "bm25_results"), ("RRF Fusion", "fused_results"), ("Reranking / Final Evidence", "reranked_results")]:
                st.markdown(f"**{label}**")
                for item in result[key]:
                    st.write(f"{item['rank']}. `{item['chunk']['chunk_id']}` · score `{item['score']:.4f}` · {item['chunk']['document_name']} p.{item['chunk']['page_number']}")
            st.json(result["validation"])
