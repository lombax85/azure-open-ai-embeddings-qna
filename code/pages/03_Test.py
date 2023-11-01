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

    st.markdown("Testo la possibilità di filtrare i documenti sui metadata prima della ricerca. In questo caso ho inserito un metadata 'permissions' e due documenti, uno salvato come registered e uno come public. Per ora sono mutualmente esclusivi")

    llm_helper = LLMHelper()

    # inizializzo la classe che si occupa degli embedings
    embeddings = OpenAIEmbeddings(
        deployment=os.getenv("OPENAI_EMBEDDINGS_ENGINE", "text-embedding-ada-002"),
        model=os.getenv('OPENAI_EMBEDDINGS_ENGINE_DOC', "text-embedding-ada-002"),
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_type="azure",
    )

    # inizializzo il vector store Redis, per comodità utilizzo le variabili che erano già presenti nel progetto nella classe llm_helper
    # sto usando RedisExtended perchè era usato nel progetto originale, ma al momento sto restituendo un'istanza di Redis
    # vedere metodo __new__ nel file code/utilities/redis.py
    # lo schema dei metadati viene inizializzato lì dentro
    vector_store: RedisExtended = RedisExtended(redis_url=llm_helper.vector_store_full_address, index_name=llm_helper.index_name, embedding_function=embeddings)  

    # Creo un filtro semplice - funziona anche con wildcard in questo caso usando % come operatore e * nella stringa
    # filters https://python.langchain.com/docs/integrations/vectorstores/redis
    filter = RedisText("permissions") % "registered"

    # inizializzo la chain
    question_generator = LLMChain(llm=llm_helper.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
    doc_chain = load_qa_with_sources_chain(llm_helper.llm, chain_type="stuff", verbose=False, prompt=llm_helper.prompt)
    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(
            search_kwargs={"filter" : filter} # inserisco i filtri come arg nel vector store
        ),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
    )

    # faccio una domanda, per ora senza history, e ottengo risultati, source e metadata
    result = chain({"question": "Chi è ICT & More?", "chat_history": {}})
    sources = "\n".join(set(map(lambda x: x.metadata["source"], result['source_documents'])))
    docmetadata = result["source_documents"]

    # stampo solo la risposta
    st.markdown("**Risposta ottenuta con query su permission registered **")
    st.markdown(result["answer"])

    # stampo i documenti che hanno portato alla risposta
    st.markdown("**Documenti utilizzati per rispondere**")
    st.markdown("\n".join(set(map(lambda x: "**File:** " + x.metadata["source"] + " **Permissions:** " + x.metadata["permissions"], result['source_documents']))))

    # st.markdown(f"Result: {result}") 
    # st.markdown(f"Sources: {sources}") 
    # st.markdown(f"Metadata: {docmetadata}") 

    st.markdown("---")

    st.markdown("**Ora proviamo con permission public**")

    filter = RedisText("permissions") % "public"
    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(
            search_kwargs={"filter" : filter} # inserisco i filtri come arg nel vector store
        ),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
    )

    # faccio una domanda, per ora senza history, e ottengo risultati, source e metadata
    result = chain({"question": "Chi è ICT & More?", "chat_history": {}})
    sources = "\n".join(set(map(lambda x: x.metadata["source"], result['source_documents'])))
    docmetadata = result["source_documents"]

    # stampo solo la risposta
    st.markdown("**Risposta ottenuta con query su permission public **")
    st.markdown(result["answer"])

    # stampo i documenti che hanno portato alla risposta
    st.markdown("**Documenti utilizzati per rispondere**")
    st.markdown("\n".join(set(map(lambda x: "**File:** " + x.metadata["source"] + " **Permissions:** " + x.metadata["permissions"], result['source_documents']))))




    st.text("end")


except Exception as e:
    st.error(traceback.format_exc())

