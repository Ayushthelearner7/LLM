from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template= 'generate a detailed report on {topic}',
    input_variables=['topic']

)
prompt2 = PromptTemplate(
    template='gernerate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']

)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'unemployment in india'})

print(result)

chain.get_graph().print_ascii()