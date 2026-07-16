from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Python is one of the most popular programming languages for AI.",
    "Embeddings convert text into numerical vector representations.",
    "LangChain is used to build applications powered by large language models.",
    "Retrieval-Augmented Generation combines retrieval systems with language models.",
    "Vector databases store and search embeddings efficiently.",
    "Computer vision enables machines to interpret images and videos.",
    "Transformers have revolutionized modern natural language processing.",
]

query  = 'What is LangChain?'

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity = list(enumerate(map(float,cosine_similarity([query_embedding], doc_embeddings)[0])))   # cosine similarity mai 2d list bhejainayz

similarity.sort(key = lambda x: x[1])

index, score = similarity[-1]

print("Query:", query)
print("Most similar document:", documents[index])
print("Similarity score:", score)

