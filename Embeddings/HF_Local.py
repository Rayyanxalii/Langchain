from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# text = 'Islaamabd is capital of Pakistan.'

# vector = embeddings.embed_query(text)

doc = [
    'Islaamabd is capital of Pakistan.',
    'New Delhi is capital of India.',
    'Beijing is capital of China.'
]

vector = embeddings.embed_documents(doc)

print(str(vector))
