from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

API_KEY=os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)


st.header("AI Research Paper Assistant")

# user_query = st.text_input("Enter your prompt")

# if st.button('Summarise'):
#     result = model.invoke(user_query)
#     st.write(result.content)

# Using dynamic prompt.

paper_input = st.selectbox("Select Research Paper Name",["Select...","Attention Is All You Need","BERT: Pre-training of Deep Biderictional Transformers","GPT-3: Language Models are few shot learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",["Begineer-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)", "Long(detailed explaination)"])

template = PromptTemplate(
    template="""
Please summarise the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details: 
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient Information available" instead of guessing.
Ensure the summary is clear, accurate and aligned with the provided style and length. 
""",
input_variables=["paper_input","style_input","length_input"]
)


prompt = template.invoke(
    {
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    }
)

if st.button('Summarise'):
    result = model.invoke(prompt)
    st.write(result.content)