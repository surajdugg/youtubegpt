#Bring in dependencies
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

#App config
st.set_page_config(page_title="Youtube Script Generator App")
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

#App framework
st.title('🎥 ⭐Youtube Script Generator ⭐ 🎥 ')
prompt = st.text_input('Choose a topic for your Youtube video')
st.text('Popular topics: Deep Learning, ChatGPT, how to take over the world...')


#create prompt template
title_template = PromptTemplate(
        input_variables = ['topic'],
        template='write me a youtube video title about {topic} '
)

script_template = PromptTemplate(
        input_variables = ['title','wikipedia_research'],
        template='write me a youtube script based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}'
)

#Memory
title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')


#LLMs
#temperature set to creative aka >0.8
llm = OpenAI(temperature=0.9)

#llm chains
title_chain = LLMChain(llm=llm, prompt= title_template, verbose = True, output_key='title',memory=title_memory)
script_chain = LLMChain(llm=llm, prompt= script_template, verbose = True, output_key='script',memory=script_memory)

#Wiki
wiki = WikipediaAPIWrapper()

#code for sequential chain --- not necessary atm
#sequential_chain = SequentialChain(chains=[title_chain,script_chain], input_variables=['topic'], output_variables=['title','script'], verbose=True)




#check for prompt and show output
if prompt:
        #response = sequential_chain({'topic':prompt})

        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        script = script_chain.run(title = title, wikipedia_research = wiki_research )
        # st.write(response['title'])
        # st.write(response['script'])

        st.write(title)
        st.write(script)

        with st.expander('Title History'):
            st.info(title_memory.buffer)

        with st.expander('Message History'):
            st.info(script_memory.buffer)

        with st.expander('Wikipedia Research'):
            st.info(wiki_research)





