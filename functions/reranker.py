from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple, Dict


class Reranker:
    def __init__(self, topk: int = 5, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.topk = int(topk)
        self.model = SentenceTransformer(model_name)

    def rerank(self, query: str, docs: List[str], metadatas: Optional[List[Dict]] = None, return_scores: bool = False) -> Tuple[List[str], Optional[List[Dict]], Optional[List[float]]]:
        # 输入验证
        if not docs:
            return [], [], [] if return_scores else ([], [], None)[2]

        n = len(docs)

        # 处理 metadatas 对齐
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        elif len(metadatas) < n:
            # 补齐缺失的 metadata
            metadatas = metadatas + [{} for _ in range(n - len(metadatas))]
        elif len(metadatas) > n:
            # 截断多余的 metadata
            metadatas = metadatas[:n]

        # 生成 embedding 并计算相似度
        try:
            query_emb = self.model.encode([query], convert_to_numpy=True)
            doc_embs = self.model.encode(docs, convert_to_numpy=True)
        except Exception as e:
            raise RuntimeError(f"Embedding 生成失败: {e}")

        # 计算余弦相似度
        scores = cosine_similarity(query_emb, doc_embs)[0]

        # 排序索引（从高到低）
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # 计算最终 topk 大小
        k = min(self.topk, len(sorted_indices))
        top_indices = sorted_indices[:k]

        # 根据索引重排 docs 和 metadatas
        sorted_docs = [docs[i] for i in top_indices]
        sorted_metadatas = [metadatas[i] for i in top_indices] if metadatas else None
        sorted_scores = [float(scores[i]) for i in top_indices]

        if return_scores:
            return sorted_docs, sorted_metadatas, sorted_scores
        else:
            return sorted_docs, sorted_metadatas, None


if __name__ == "__main__":
    reranker = Reranker()
    query = "什么是RAG？"
    docs = [
        "早餐就要吃包子",
        "RAG代表检索增强生成，是一种结合了信息检索和生成模型的技术。",
        "机器学习是一种人工智能的分支。",
        "Python是一种流行的编程语言。"
    ]
    metadatas = [
        {"source": "doc1.txt"},
        {"source": "doc2.txt"},
        {"source": "doc3.txt"},
        {"source": "doc4.txt"}
    ]
    
    ranked_docs, ranked_metadatas, scores = reranker.rerank(query, docs, metadatas, return_scores=True)
    for i, (doc, meta, score) in enumerate(zip(ranked_docs, ranked_metadatas, scores)):
        print(f"Rank {i+1}: {doc} | Metadata: {meta} | Score: {score}")
