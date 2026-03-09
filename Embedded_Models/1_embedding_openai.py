from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

Embedding = OpenAIEmbeddings(model='gemini-embedding-001' , dimensions=32)    

result = Embedding.embed_query("delhi is the capital of india")

print(str(result))
