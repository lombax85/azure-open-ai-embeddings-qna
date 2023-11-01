import streamlit as st
import os
import traceback
from utilities.helper import LLMHelper
import streamlit.components.v1 as components
from urllib import parse

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores.redis.filters import RedisFilterExpression, RedisText, RedisTag

from utilities.redis import RedisExtended

try:
    # Set page layout to wide screen and menu item
    menu_items = {
	'Get help': None,
	'Report a bug': None,
	'About': '''
	 ## Embeddings App

	Document Reader Sample Demo.
	'''
    }
    st.set_page_config(layout="wide", menu_items=menu_items)

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


    llm_helper = LLMHelper()

    st.text("start")

    # Create LLMHelper object
    llm_helper = LLMHelper()
    # Get answer

    answer = llm_helper.get_semantic_answer_lang_chain_updated("chi è ict & more?", {})
    data = {}
    data['question'], data['response'], data['context'], data["sources"] = answer
    st.markdown(answer)
    st.text("end")


except Exception as e:
    st.error(traceback.format_exc())

