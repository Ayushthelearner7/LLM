from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "delhi is the capital of india",
    "patna is the capital of bihar",
    "hello my name is ayush raj"
]

vector = embedding.embed_documents(documents)

print(str(vector))
