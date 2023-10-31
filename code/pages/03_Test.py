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

from utilities.redis2 import RedisExtended

from sys import settrace

# local trace function which returns itself 
def my_tracer(frame, event, arg = None): 
    st.text(frame)
    # extracts frame code 
    code = frame.f_code 

    # extracts calling function name 
    func_name = code.co_name 
  
    # extracts the line number 
    line_no = frame.f_lineno 
  
    st.text(f"A {event} encountered in {func_name}() at line number {line_no} ") 
  
    return my_tracer 


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

    st.text("test page")


    llm_helper = LLMHelper()


    # answer = llm_helper.get_semantic_answer_lang_chain(
    #     "chi è ict & more",
    #     ""
    # )
    # st.text(answer)

    # vector_store: RedisExtended = RedisExtended(redis_url=llm_helper.vector_store_full_address, index_name=llm_helper.index_name, embedding_function=llm_helper.embeddings.embed_query)  
    vector_store: RedisExtended = RedisExtended(redis_url=llm_helper.vector_store_full_address, index_name=llm_helper.index_name, embedding_function=llm_helper.embeddings.embed_query)  

    # st.text(vector_store.__class__.__name__)
    # st.text(vector_store.as_retriever().__class__.__name__)


    question_generator = LLMChain(llm=llm_helper.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
    doc_chain = load_qa_with_sources_chain(llm_helper.llm, chain_type="stuff", verbose=False, prompt=llm_helper.prompt)
    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        # top_k_docs_for_context= self.k
    )
    # settrace(my_tracer)
    result = chain({"question": "Chi è ICT & More?", "chat_history": {}})
    sources = "\n".join(set(map(lambda x: x.metadata["source"], result['source_documents'])))
    docmetadata = result["source_documents"]

    st.markdown(f"Result: {result}") 
    st.markdown(f"Sources: {sources}") 
    st.markdown(f"Metadata: {docmetadata}") 

    st.text("end")


except Exception as e:
    st.error(traceback.format_exc())

