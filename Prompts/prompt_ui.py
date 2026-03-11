import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt

load_dotenv()

# 1. Setup the Free Hugging Face Endpoint
# This replaces the paid ChatOpenAI()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)

# 2. Wrap it in ChatHuggingFace for ChatPromptTemplate compatibility
model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox( 
    "Select Research Paper Name", 
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] 
)

style_input = st.selectbox( 
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] 
) 

length_input = st.selectbox( 
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] 
)

# Ensure template.json exists in your directory
try:
    template = load_prompt('template.json')
except Exception as e:
    st.error("Make sure 'template.json' is in your folder!")
    st.stop()

if st.button('Summarize'):
    with st.spinner('Generating summary...'):
        chain = template | model
        result = chain.invoke({
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
        st.write(result.content)