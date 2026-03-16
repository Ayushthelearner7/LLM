import os
import traceback
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ── 1. Validate Token ─────────────────────────────────────────────────────────
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file!")

# ── 2. HuggingFace Client (replaces ChatOpenAI) ───────────────────────────────
client = InferenceClient(token=token)

def call_llm(prompt_value) -> str:
    prompt_str = str(prompt_value)
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt_str}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content

# Wrap as LangChain Runnable (replaces ChatOpenAI model object)
model = RunnableLambda(call_llm)

# ── 3. Prompt Templates (same as your OpenAI code) ────────────────────────────

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# ── 4. Output Parser ──────────────────────────────────────────────────────────
parser = StrOutputParser()

# ── 5. Chain (exact same structure as your OpenAI code) ───────────────────────
# template1 | model | parser  →  gives detailed report as string
# that string goes into template2 as {text}
# template2 | model | parser  →  gives final 5 line summary

chain = (
    template1
    | model
    | parser
    | (lambda text: {"text": text})   # map string output to template2's input variable
    | template2
    | model
    | parser
)

# ── 6. Run ────────────────────────────────────────────────────────────────────
result = chain.invoke({'topic': 'black hole'})
print(result)