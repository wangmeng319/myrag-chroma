from functions import reranker, retriever, prompt_builder, build_index, llm_client
import dotenv
dotenv.load_dotenv()

def demo():
    retriever_instance = retriever.ChromaRetriever(k=10, similarity_threshold=0.3)
    reranker_instance = reranker.Reranker(topk=3)
    llm = llm_client.LLMClient()
    pb = prompt_builder.PromptBuilder()

    question = input("请输入您的问题: ")
    while question.lower() != "exit" and question.lower() != "quit":
        # 检索相关文档
        docs, metadatas, _ = retriever_instance.retrieve(question)

        if not docs:
            print("未找到相关文档。")
        else:
            # 重新排序文档
            ranked_docs, ranked_metadatas, _ = reranker_instance.rerank(question, docs, metadatas)

            # 构建提示词
            context = ""
            for i, (doc, meta) in enumerate(zip(ranked_docs, ranked_metadatas)):
                print(f"Rank {i+1}: {doc} | Metadata: {meta}")
                context += f"文档{i+1}:\n来源: {meta['source']}\n内容: {doc}\n\n"
            prompt = pb(context, question)

            # 生成回答
            answer = llm.generate(prompt)

            print("用户问题:", question)
            print("生成回答:", answer)

        question = input("\n请输入您的问题 (或输入 'exit' 退出): ")


def main():
    # 构建索引
    # build_index.build_index()

    # 初始化组件
    retriever_instance = retriever.ChromaRetriever(k=3)
    reranker_instance = reranker.Reranker()
    llm = llm_client.LLMClient()

    # 用户问题
    question = "什么是RAG？"

    # 检索相关文档
    docs, metadatas = retriever_instance.retrieve(question)

    # 重新排序文档
    ranked_docs, ranked_metadatas, _ = reranker_instance.rerank(question, docs, metadatas)

    # 构建提示词
    context = ""
    for i, (doc, meta) in enumerate(zip(ranked_docs, ranked_metadatas)):
        print(f"Rank {i+1}: {doc} | Metadata: {meta}")
        context += f"文档{i+1}:\n来源: {meta['source']}\n内容: {doc}\n\n"
    pb = prompt_builder.PromptBuilder()
    prompt = pb(context, question)
    print("构建的提示词:")
    print(prompt)

    # # 生成回答
    # answer = llm.generate(prompt)

    # print("用户问题:", question)
    # print("生成回答:", answer)

if __name__ == "__main__":
    demo()

