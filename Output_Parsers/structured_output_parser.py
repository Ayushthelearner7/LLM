from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()

import os
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file!")

# ── Model (replaces gemma with Llama already working on your account) ──────────
client = InferenceClient(token=token)

def call_llm(prompt_value) -> str:
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": str(prompt_value)}],
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

model = RunnableLambda(call_llm)

# ── Schema (replaces ResponseSchema + StructuredOutputParser) ─────────────────
class FactSchema(BaseModel):
    fact_1: str = Field(description='Fact 1 about the topic')
    fact_2: str = Field(description='Fact 2 about the topic')
    fact_3: str = Field(description='Fact 3 about the topic')

parser = JsonOutputParser(pydantic_object=FactSchema)

# ── Prompt Template (same as your original) ───────────────────────────────────
template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# ── Chain (same structure as your original) ───────────────────────────────────
chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)