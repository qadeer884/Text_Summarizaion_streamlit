import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Set API Key (replace with your actual key)
os.environ["GROQ_API_KEY"] = "Enter a groq api key here "

# Load the language model
model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5
)

# Define text splitter
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100
)

# Summarization and Refinement Prompts
messages_summarize = [
    ("system", "You are an advanced AI summarization assistant. Your task is to summarize a given text without losing important details. The summary should be approximately one-fourth of the original text length."),
    ("human", "Summarize the following text in a concise manner while preserving key points and just give summary and no other context in the summary like 'here is concise summary' etc:\n\n{text}")
]

messages_refine = [
    ("system", "You are an advanced AI summarization assistant. Your task is to refine a given summary for readability and coherence without altering key points or introducing redundancy."),
    ("human", "Refine the following summary for better flow and clarity, ensuring it remains concise and non-repetitive:\n\n{text}")
]

# Create Prompt Templates
prompt_template_summarize = ChatPromptTemplate.from_messages(messages_summarize)
prompt_template_refine = ChatPromptTemplate.from_messages(messages_refine)

# Streamlit UI
st.title("AI-Powered Text Summarization with LLaMA & Groq")
st.write("Enter a text below, and our AI will generate a concise and meaningful summary!")

text_input = st.text_area("Paste your text here:", height=200)

if st.button("Summarize"):
    if text_input:
        split_text = r_splitter.split_text(text_input)
        chain_summarize = prompt_template_summarize | model | StrOutputParser()
        
        if len(text_input) < 2000:
            final_summary = chain_summarize.invoke({"text": text_input})
        else:
            summaries = [chain_summarize.invoke({"text": chunk}) for chunk in split_text]
            beta_summary = " ".join(summaries)
            
            chain_refine = prompt_template_refine | model | StrOutputParser()
            final_summary = chain_refine.invoke({"text": beta_summary})
        
        st.subheader("Summarized Text")
        st.write(final_summary)
    else:
        st.warning("Please enter text to summarize.")
