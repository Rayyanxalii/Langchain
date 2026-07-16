from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

LLM  = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V4-Pro',
    task= 'text-generation'
)

model = ChatHuggingFace(
    llm = LLM
)

response = model.invoke('What is the capital of Pakistan?')
print(response.content)