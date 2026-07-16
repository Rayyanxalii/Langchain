from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings  = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)

embedding_vector = embeddings.embed_query("What is the capital of Pakistan?")

print(str(embedding_vector))


# for documents

# documents = ["What is the capital of Pakistan?", "What is the capital of India?", "What is the capital of China?"]

# embedding_vector = embeddings.embed_documents(documents)
