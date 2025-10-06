from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os
import dotenv
import numpy as np
dotenv.load_dotenv()

PERSIST_DIRECTORY = os.getenv("CHROMA_DB_DIR", "data/index/chroma_db")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)


class ChromaRetriever:
    def __init__(self, k: int = 5, similarity_threshold: float = 0.3):
        self.client = PersistentClient(path=PERSIST_DIRECTORY)
        self.collection = self.client.get_collection("rag_docs")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.k = k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str):
        # 先从 Chroma 中获取候选（最多 self.k）
        # 注意：collection.query 结果是嵌套的（按查询），所以取 [0]
        try:
            # request documents and metadatas
            results = self.collection.query(
                query_embeddings=self.model.encode([query]).tolist(),
                n_results=self.k,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            raise RuntimeError(f"从 Chroma 检索失败: {e}")

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not docs:
            return [], [], []

        # 计算实际的余弦相似度以确保一致的度量并进行阈值过滤
        try:
            q_emb = self.model.encode([query], convert_to_numpy=True)
            doc_embs = self.model.encode(docs, convert_to_numpy=True)
        except Exception as e:
            raise RuntimeError(f"生成 embedding 失败: {e}")

        q = q_emb[0]
        # 防止除零
        q_norm = np.linalg.norm(q)
        doc_norms = np.linalg.norm(doc_embs, axis=1)
        # 计算余弦相似度
        sims = []
        for i, d in enumerate(doc_embs):
            denom = (q_norm * doc_norms[i])
            sim = float(np.dot(q, d) / denom) if denom != 0 else 0.0
            sims.append(sim)

        # 过滤并按相似度降序排序
        filtered = [(i, sims[i]) for i in range(len(sims)) if sims[i] >= self.similarity_threshold]
        if not filtered:
            return [], [], []

        # sort by similarity desc
        filtered.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, s in filtered]

        filtered_docs = [docs[i] for i in indices]
        filtered_metas = [metadatas[i] for i in indices]
        filtered_scores = [s for _, s in filtered]

        return filtered_docs, filtered_metas, filtered_scores


if __name__ == "__main__":
    retriever = ChromaRetriever(k=10)
    query = "什么是高血压？"
    docs, metadatas, scores = retriever.retrieve(query)
    for i, (doc, meta, score) in enumerate(zip(docs, metadatas, scores)):
        print(f"Document {i+1} (score={score:.4f}):")
        print(f"Source: {meta.get('source')}")
        print(f"Content: {doc}\n")