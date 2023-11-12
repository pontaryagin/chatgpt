#!/usr/bin/env python3
import pysqlite_load
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="./storage", embedding_function=embeddings)
retriever = db.as_retriever()

model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=500)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

question = "please summarize the document"

while True:
	print("Q:" + (question or ""))
	query = question or input()
	question = None
	answer = qa.run(query)
	print("A:", answer)
