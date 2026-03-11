from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# 1. Setup the free model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
)

# 2. Wrap it for Chat Template support
model = ChatHuggingFace(llm=llm)

# 3. Define the Chat Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# 4. Load chat history from your file
chat_history = []
try:
    with open('chat_history.txt', 'r') as f:
        # We strip newlines to keep the list clean
        chat_history.extend([line.strip() for line in f.readlines()])
except FileNotFoundError:
    print("chat_history.txt not found, starting with empty history.")

# 5. Invoke the template
# Note: LangChain expects chat_history to be a list of message objects or strings
prompt = chat_template.invoke({
    'chat_history': chat_history, 
    'query': 'Where is my refund'
})

# 6. Get the response
result = model.invoke(prompt)
print("AI Response:", result.content)