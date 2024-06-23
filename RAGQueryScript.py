from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import sys
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# caching previous responses in LangChain SQLLite db
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH")

RAG_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Answer the question based on the above context: {question}
"""

query_text = sys.argv[1]

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

#Number of chunks to sample
k = 4

# Search the DB.
results = db.similarity_search_with_relevance_scores(query_text, k=k)
if len(results) == 0 or results[0][1] < 0.66:
    print(f"Unable to find matching results.")
    exit()

context_input = "\n\n".join([a[0].page_content for a in results])
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE).format(context=context_input, question=query_text)

model = ChatOpenAI()
response_text = model.invoke(rag_prompt)
sources = [doc.metadata.get("source", None) for doc, _score in results]
print(f"Response from OpenAI: {response_text.content}\n")
print(f"These were the {k} relevant chunk(s) provided:\n {context_input}\n\n")