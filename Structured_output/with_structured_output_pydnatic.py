from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# 1. Setup the Free Hugging Face Model
# Llama-3.1-8B is excellent at structured JSON output
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    return_full_text=False,
)
model = ChatHuggingFace(llm=llm)

# 2. Use your existing Pydantic Schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# 3. Setup Parser and Prompt
parser = JsonOutputParser(pydantic_object=Review)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert data extractor. Extract the information into JSON format.\n{format_instructions}"),
    ("human", "{review_text}")
])

# 4. Build the Chain
chain = prompt | model | parser

# 5. Execute
review_content = """I recently upgraded to the Samsung Galaxy S24 Ultra... [Your Review Text] ... Review by Nitish Singh"""

result = chain.invoke({
    "review_text": review_content,
    "format_instructions": parser.get_format_instructions()
})

print(result)