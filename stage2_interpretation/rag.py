"""A true (vector) RAG over the knowledge corpus: embed chunks with a local embedding model, retrieve the
most relevant ones for a given tongue via cosine similarity (faiss). This ADDS retrieved, cited context to
the LLM's grounding beyond the per-feature structured facts — so it can reason about combinations and
disambiguations it wasn't hand-fed.

Build the index once:
    python stage2_interpretation/rag.py --build
Then at inference the Retriever loads the cached vectors (no re-embed).

Embeddings via the local Ollama endpoint (nomic-embed-text, no auth). Falls back gracefully: if the
embedder or index is unavailable, the caller just gets no retrieved context (structured grounding remains).
"""
import json, os, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "knowledge_base", "corpus.jsonl")
VECS = os.path.join(HERE, "knowledge_base", "corpus_vecs.npy")
META = os.path.join(HERE, "knowledge_base", "corpus_meta.json")
EMBED_URL = os.getenv("TIH_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.getenv("TIH_EMBED_MODEL", "nomic-embed-text")


def embed(text, timeout=20):
    """Return a normalized embedding vector for `text`, or None on failure."""
    try:
        req = urllib.request.Request(EMBED_URL, data=json.dumps({"model": EMBED_MODEL, "prompt": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        v = np.array(json.loads(urllib.request.urlopen(req, timeout=timeout).read())["embedding"], np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None
    except Exception:
        return None


def build_index():
    chunks = [json.loads(l) for l in open(CORPUS) if l.strip()]
    vecs, meta = [], []
    for c in chunks:
        v = embed(c["text"])
        if v is None:
            print("WARN: embed failed for", c["id"]); continue
        vecs.append(v); meta.append(c)
    arr = np.vstack(vecs).astype(np.float32)
    np.save(VECS, arr)
    json.dump(meta, open(META, "w"), ensure_ascii=False)
    print(f"built index: {len(meta)} chunks x {arr.shape[1]}d -> {VECS}")


class Retriever:
    """Loads the cached index; retrieve(query, k) -> list of {id, text, source, score}. No-op if the index
    or embedder is missing (returns [])."""
    def __init__(self):
        self.ok = os.path.exists(VECS) and os.path.exists(META)
        if self.ok:
            self.vecs = np.load(VECS)
            self.meta = json.load(open(META))
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.vecs.shape[1])   # cosine (vectors are normalized)
                self.index.add(self.vecs)
            except Exception:
                self.index = None

    def retrieve(self, query, k=5, min_score=0.35):
        if not self.ok:
            return []
        q = embed(query)
        if q is None:
            return []
        if self.index is not None:
            D, I = self.index.search(q[None, :], k)
            hits = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        else:
            sims = self.vecs @ q
            idx = np.argsort(-sims)[:k]
            hits = [(int(i), float(sims[i])) for i in idx]
        out = []
        for i, s in hits:
            if s < min_score:
                continue
            m = self.meta[i]
            out.append({"id": m["id"], "text": m["text"], "source": m.get("source", ""), "score": round(s, 3)})
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None)
    args = ap.parse_args()
    if args.build:
        build_index()
    if args.query:
        for h in Retriever().retrieve(args.query, k=5):
            print(f"[{h['score']:.2f}] {h['id']}: {h['text'][:110]}")
