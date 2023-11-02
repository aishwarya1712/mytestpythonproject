# get dependencies
import os 
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI 

os.environ['OPENAI_API_KEY'] = apikey # dictionary

# app framework 
st.title("🦜️🔗 Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

# LLMS
llm = OpenAI(temperature=0.9)

# trigger prompt to LLM
if(prompt):
    response = llm(prompt) # gets response from LLM
    st.write(response) # displays on screen