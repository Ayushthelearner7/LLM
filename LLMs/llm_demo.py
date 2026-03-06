from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# This is the "Parallel" setup. 
# Your tutor might just have llm = ChatOpenAI()
# You just add these 3 specific lines inside the parentheses:
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyA1LnRz78C-xzuZiIRxr-O0mcx9wqcs33g", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

result = llm.invoke("What is the capital of India?")

print(result.content)