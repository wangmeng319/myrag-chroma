import os, dotenv


dotenv.load_dotenv()

template = os.getenv("PROMPT_TEMPLATE", """基于以下文档内容，简要回答用户的问题。如果文档内容无法回答问题，请基于已有知识进行回答，不要编造信息。
文档内容:   {context}
用户问题: {question}
回答:""")

class PromptBuilder:
    def __init__(self, template=template):
        self.template = template

    def __call__(self, context, question):
        return self.template.format(context=context, question=question)
    

if __name__ == "__main__":
    prompt_builder = PromptBuilder()
    context = "RAG代表检索增强生成，是一种结合了信息检索和生成模型的技术。"
    question = "什么是RAG？"
    prompt = prompt_builder(context, question)
    print(prompt)