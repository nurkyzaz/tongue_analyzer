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
    """HYBRID retriever: fuses semantic (embeddings) + lexical (TF-IDF) ranks via Reciprocal Rank Fusion.
    Semantic captures meaning; lexical catches exact terms ('peeled', 'cracks', 'tooth-marks') that a
    coating-heavy query would otherwise let dominate. retrieve(query,k) -> [{id,text,source,score}].
    Degrades: no embedder -> lexical-only (works offline); no index -> []."""
    def __init__(self):
        self.ok = os.path.exists(VECS) and os.path.exists(META)
        if not self.ok:
            return
        self.vecs = np.load(VECS)
        self.meta = json.load(open(META))
        # lexical index over the SAME chunks (refit at load; corpus is small). id text helps exact terms.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus_txt = [m["id"].split(":", 1)[-1].replace("_", " ") + ". " + m["text"] for m in self.meta]
            self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            self.lex = self.tfidf.fit_transform(corpus_txt)
        except Exception:
            self.tfidf = None

    @staticmethod
    def _ranks(scores, n):
        order = np.argsort(-scores)[:n]
        return {int(i): r for r, i in enumerate(order)}

    def retrieve(self, query, k=5, pool=15, rrf=60):
        if not self.ok:
            return []
        sem_rank, lex_rank = {}, {}
        q = embed(query)
        if q is not None:
            sem_rank = self._ranks(self.vecs @ q, pool)
        if self.tfidf is not None:
            lex_scores = (self.lex @ self.tfidf.transform([query]).T).toarray().ravel()
            lex_rank = self._ranks(lex_scores, pool)
        if not sem_rank and not lex_rank:
            return []
        # Reciprocal Rank Fusion across whichever lists we have
        fused = {}
        for ranks in (sem_rank, lex_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf + r)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        out = []
        for i in top:
            m = self.meta[i]
            out.append({"id": m["id"], "text": m["text"], "source": m.get("source", ""),
                        "license": m.get("license", "own"), "usage": m.get("usage", "authored-summary"),
                        "type": m.get("type", "card"), "tags": m.get("tags", []),
                        "score": round(fused[i], 4)})
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
