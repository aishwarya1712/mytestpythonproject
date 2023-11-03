# get dependencies
import os 
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI 

from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

os.environ['OPENAI_API_KEY'] = apikey # dictionary

# app framework 
st.title("🦜️🔗 Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

# create a prompt template
# create a topic and write something about it
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me a youtube video title about {topic}'
)

# generated YT video script 
script_template = PromptTemplate(
    input_variables = ['title'],
    template = 'Write me a youtube video script based on this title: {title}'
)


# LLMS
llm = OpenAI(temperature=0.9)

# ----- Simple Sequential Chain --------
# to use the above templates, we need to have an LLM chain
# title_chain = LLMChain(llm=llm, prompt=title_template,verbose=True)
# script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

# connect the chains together
# the output of the first goes into the input of the second - simple sequential chain
# seq_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True) # order is important here!! 

# the simple sequential chain only displays the output of the LAST output of the llm chain.
# to grab multiple outputs, it gets trickier.
# we can use SequentialChain instead of SSQ to get multiple outputs.
#seq_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True) # order is important here!! 

# trigger prompt to LLM - For SSQ
# if(prompt):
    # response = seq_chain.run(prompt) # gets response from LLM for SSQ
    # st.write(response) # displays on screen

# ----- Sequential Chain --------
# to use the above templates, we need to have an LLM chain
title_chain = LLMChain(llm=llm, prompt=title_template,verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')

# connect the chains together
# the output of the first goes into the input of the second - sequential chain
seq_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title','script'], verbose=True) # order is important here!! 

# trigger prompt to LLM
if(prompt):
    response = seq_chain({'topic':prompt})
    st.write(response['title']) # displays on screen
    st.write(response['script']) # displays on screen


