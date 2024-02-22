import azure.functions
from dotenv import load_dotenv
load_dotenv()

import os
from utilities.helper import LLMHelper
from langchain.embeddings.openai import OpenAIEmbeddings

def main(req: azure.functions.HttpRequest) -> str:
    custom_prompt = ""
    custom_temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
    # Create LLMHelper object
    llm_helper = LLMHelper(custom_prompt=custom_prompt, temperature=custom_temperature)
    return f'{llm_helper.streaming()}'