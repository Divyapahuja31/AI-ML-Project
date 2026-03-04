"""
Retrieval-Augmented Generation (RAG) module.

Embeds lending policy text documents into a FAISS vector index using
``sentence-transformers`` and provides a semantic search interface.

Policy documents are stored as plain-text files in ``data/policies/``.
The built index is cached to ``models/rag_index.pkl`` so that the
embedding step only runs once.
"""

import os
import pickle
from typing import Dict, List

import numpy as np

POLICIES_DIR   = "data/policies"
RAG_INDEX_PATH = "models/rag_index.pkl"
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHUNK_MIN_LEN  = 80  # Minimum characters for a chunk to be indexed


# ─── Document loading ─────────────────────────────────────────────────────────

def _chunk_document(text: str, source: str) -> List[Dict]:
    """Split a document string into paragraph-level chunks."""
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) >= CHUNK_MIN_LEN:
            chunks.append({"source": source, "content": para})
    return chunks


def load_policy_documents() -> List[Dict]:
    """Load all ``.txt`` files from :data:`POLICIES_DIR` as chunked dicts."""
    docs = []
    if not os.path.isdir(POLICIES_DIR):
        return docs

    for fname in sorted(os.listdir(POLICIES_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(POLICIES_DIR, fname)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            docs.extend(_chunk_document(content, fname.replace(".txt", "")))

    return docs


# ─── Index construction ───────────────────────────────────────────────────────

def build_rag_index() -> Dict:
    """Build and persist a FAISS index from policy documents.

    Returns
    -------
    dict
        ``{"index": faiss.Index, "docs": list}``
    """
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore

    docs = load_policy_documents()
    if not docs:
        raise ValueError(f"No policy documents found in '{POLICIES_DIR}'.")

    print(f"⏳  Embedding {len(docs)} policy chunks …")
    encoder    = SentenceTransformer(EMBED_MODEL)
    embeddings = encoder.encode(
        [d["content"] for d in docs],
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    artifact = {"index": index, "docs": docs}
    os.makedirs("models", exist_ok=True)
    with open(RAG_INDEX_PATH, "wb") as fh:
        pickle.dump(artifact, fh)

    print(f"✅  RAG index built ({len(docs)} chunks) → '{RAG_INDEX_PATH}'")
    return artifact


def _load_or_build() -> Dict:
    if os.path.exists(RAG_INDEX_PATH):
        with open(RAG_INDEX_PATH, "rb") as fh:
            return pickle.load(fh)
    return build_rag_index()


# ─── Public query API ─────────────────────────────────────────────────────────

def query_policies(query: str, k: int = 4) -> List[Dict]:
    """Retrieve the *k* most relevant policy chunks for *query*.

    Parameters
    ----------
    query : str
        Natural-language search string.
    k : int
        Number of results to return.

    Returns
    -------
    list of dicts with keys ``source`` and ``content``.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore

    artifact  = _load_or_build()
    encoder   = SentenceTransformer(EMBED_MODEL)
    query_emb = encoder.encode([query], convert_to_numpy=True)

    _, indices = artifact["index"].search(query_emb.astype(np.float32), k)
    return [
        artifact["docs"][i]
        for i in indices[0]
        if i < len(artifact["docs"])
    ]


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_rag_index()
    results = query_policies("high risk borrower with 90 days late payment", k=3)
    for r in results:
        print(f"\n[{r['source']}]\n{r['content'][:300]}")
