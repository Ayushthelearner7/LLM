from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the free Hugging Face model
# This runs locally on your computer (no API key required)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about who is god'

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate similarity scores
# cosine_similarity expects 2D arrays, so we wrap query_embedding in brackets
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the document with the highest score
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print("index is " ,index)
print(f"Best Match: {documents[index]}")
print(f"Similarity Score: {score:.4f}")