import json
import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from langchain.vectorstores.redis import Redis

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

from langchain.schema import BaseRetriever
from pydantic import BaseModel, Extra, Field, root_validator

import pandas as pd
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TagField, TextField

import streamlit as st

from langchain.vectorstores.redis.filters import RedisFilterExpression, RedisText, RedisTag


logger = logging.getLogger()
class RedisVectorStoreRetriever(VectorStoreRetriever):
    def __init__(
        self,
        vectorstore: Any,
        **kwargs: Any, 
    ):
        st.text("vectorestore is")
        st.text(vectorstore)
        super().__init__(vectorstore=vectorstore)
        for key, value in kwargs.items():
            st.text("setting item")
            st.text(value)
            setattr(self, 'search_kwargs', value)
    """Retriever for Redis VectorStore."""

    vectorstore: Redis
    """Redis VectorStore."""
    search_type: str = "similarity"
    """Type of search to perform. Can be either
    'similarity',
    'similarity_distance_threshold',
    'similarity_score_threshold'
    """

    search_kwargs: Dict[str, Any] = {
        "k": 4,
        "score_threshold": 0.9,
        # set to None to avoid distance used in score_threshold search
        "distance_threshold": None,
    }
    """Default search kwargs."""

    allowed_search_types = [
        "similarity",
        "similarity_distance_threshold",
        "similarity_score_threshold",
        "mmr",
    ]
    """Allowed search types."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> List[Document]:
        if self.search_type == "similarity":
            st.markdown(self.search_kwargs)
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
            st.markdown(docs)
        elif self.search_type == "similarity_distance_threshold":
            if self.search_kwargs["distance_threshold"] is None:
                raise ValueError(
                    "distance_threshold must be provided for "
                    + "similarity_distance_threshold retriever"
                )
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)

class RedisExtended(Redis):
    def __init__(
        self,
        redis_url: str,
        index_name: str,
        embedding_function: Callable,
        **kwargs: Any,
    ):
        schema = {
            "text": [
                {"name": "source"},
                {"name": "key"},
                {"name": "chunk"},
                {"name": "filename"},
                {"name": "test_meta"},
            ]
        }
        super().__init__(redis_url, index_name, embedding_function, index_schema=schema)

        st.text("prova123")

        for key, value in kwargs.items():
            st.text(value)
            setattr(self, key, value)

        # Check if index exists
        try:
            self.client.ft("prompt-index").info()
        except: 
            # Create Redis Index
            self.create_prompt_index()

        try:
            self.client.ft(self.index_name).info()
        except:
            # Create Redis Index
            self.create_index()

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        filter = RedisText("test_meta") % "t*"
        return RedisVectorStoreRetriever(vectorstore=self, search_kwargs={'filter': filter})

    def check_existing_index(self, index_name: str = None):
        try:
            self.client.ft(index_name if index_name else self.index_name).info()
            return True
        except:
            return False

    def delete_keys(self, keys: List[str]) -> None:
        for key in keys:
            self.client.delete(key)
    
    def delete_keys_pattern(self, pattern: str) -> None:
        keys = self.client.keys(pattern)
        self.delete_keys(keys)

    def create_index(self, prefix = "doc", distance_metric:str="COSINE"):
        content = TextField(name="content")
        metadata = TextField(name="metadata")
        content_vector = VectorField("content_vector",
                    "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": 1536,
                        "DISTANCE_METRIC": distance_metric,
                        "INITIAL_CAP": 1000,
                    })
        # Create index
        self.client.ft(self.index_name).create_index(
            fields = [content, metadata, content_vector],
            definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        )

    # Prompt management
    def create_prompt_index(self, index_name="prompt-index", prefix = "prompt"):
        result = TextField(name="result")
        filename = TextField(name="filename")
        prompt = TextField(name="prompt")
        # Create index
        self.client.ft(index_name).create_index(
            fields = [result, filename, prompt],
            definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        )

    def add_prompt_result(self, id, result, filename="", prompt=""):
        self.client.hset(
            f"prompt:{id}",
            mapping={
                "result": result,
                "filename": filename,
                "prompt": prompt
            }
        )

    def get_prompt_results(self, prompt_index_name="prompt-index", number_of_results: int=3155):
        base_query = f'*'
        return_fields = ['id','result','filename','prompt']
        query = Query(base_query)\
            .paging(0, number_of_results)\
            .return_fields(*return_fields)\
            .dialect(2)
        results = self.client.ft(prompt_index_name).search(query)
        if results.docs:
            return pd.DataFrame(list(map(lambda x: {'id' : x.id, 'filename': x.filename, 'prompt': x.prompt, 'result': x.result.replace('\n',' ').replace('\r',' '),}, results.docs))).sort_values(by='id')
        else:
            return pd.DataFrame()

    def delete_prompt_results(self, prefix="prompt*"):
        self.delete_keys_pattern(pattern=prefix)


