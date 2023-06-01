#Bring in dependencies
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

#App framework
st.title('🎥 Youtube GPT Creator')
st.caption('Enter a topic and get an effective video title with topics that you can cover in your video')
prompt = st.text_input('Pick a topic you want to talk about')


#create prompt template
title_template = PromptTemplate(
        input_variables = ['topic'],
        template='Give me an attention grabbing headlines for a youtube video about {topic}'
)

script_template = PromptTemplate(
        input_variables = ['title'],
        template='Give me bullet points of what I can talk about in a youtube video based on this title TITLE: {title}'
)

#Memory
memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')

#LLMs
#temperature set to creative aka >0.8
llm = OpenAI(temperature=0.9)

#llm chains
title_chain = LLMChain(llm=llm, prompt= title_template, verbose = True, output_key='title',memory=memory)
script_chain = LLMChain(llm=llm, prompt= script_template, verbose = True, output_key='script',memory=memory)
sequential_chain = SequentialChain(chains=[title_chain,script_chain], input_variables=['topic'], output_variables=['title','script'], verbose=True)




#check for prompt and show output
if prompt:
        response = sequential_chain({'topic':prompt})
        st.markdown('**Here are some possible titles:** ')
        st.write(response['title'])
        st.markdown('**Here is what you can talk about:**')
        st.write(response['script'])

        with st.expander('Message History'):
            st.info(memory.buffer)






