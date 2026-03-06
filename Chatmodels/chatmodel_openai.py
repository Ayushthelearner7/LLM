from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyA1LnRz78C-xzuZiIRxr-O0mcx9wqcs33g", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=1.5,
    max_completion_tokens=5000
)

result = model.invoke("write 5 line poem on cricket")

print(result.content)