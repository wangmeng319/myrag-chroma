import os
import dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from pathlib import Path


dotenv.load_dotenv(dotenv.find_dotenv())

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 360))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    "将文本分割成chunk"
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


data_path = os.getenv("DATA_PATH", "data/docs")
persist_directory = os.getenv("CHROMA_DB_DIR", "data/index/chroma")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")


def build_index(docs_path=data_path, persist_directory=persist_directory):
    os.makedirs(persist_directory, exist_ok=True)

    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="rag_docs")

    model = SentenceTransformer(embedding_model_name)

    docs, metadatas, ids = [], [], []
    idx = 0

    for path in Path(docs_path).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            docs.append(chunk)
            metadatas.append({"source": str(path)})
            ids.append(f"doc_{idx}")
            idx += 1

    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)

    print(f"索引已保存到 {persist_directory}")



if __name__ == "__main__":
    build_index()