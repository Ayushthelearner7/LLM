from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", # Example of a supported model
    task="text-generation",
    provider="auto", # Tells HF to find ANY available server
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of bihar state")

print(result.content)