from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Setup the Free Hugging Face Model
# We use Llama-3.1 because it handles chat history very well
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)

model = ChatHuggingFace(llm=llm)

# Initialize Chat History
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

print("--- Chat Started (Type 'exit' to stop) ---")

while True:
    user_input = input('You: ')
    
    if user_input.lower() == 'exit':
        break
        
    # Add User Message to History
    chat_history.append(HumanMessage(content=user_input))
    
    # Get Response from Hugging Face
    result = model.invoke(chat_history)
    
    # Add AI Response to History so it "remembers" previous context
    chat_history.append(AIMessage(content=result.content))
    
    print(f"AI: {result.content}")

print("\n--- Final Chat History Log ---")
print(chat_history)