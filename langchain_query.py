#!/usr/bin/env python3
import pysqlite_load
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("./storage", embeddings)
retriever = vectorstore.as_retriever(
	search_type="mmr",
	search_kwargs={'k': 6, 'lambda_mult': 0.25}
	)

model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=500)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
								retriever=retriever,
								return_source_documents=True,
								chain_type_kwargs=dict(verbose=True))

question = "please summarize the document"


# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template, verbose=True)

# model = ChatOpenAI()

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
# )

while True:
	print("Q:" + (question or ""))
	query = question or input()
	question = None
	# answer = qa.run(query)
	res = qa({"query": query})
	res["source_documents"][0]
	print(f"Answear refering the following data.")
	doc: Document
	for doc in res['source_documents']:
		print(f"[Page {doc.metadata.get('page')}] {doc.page_content}")
	print("A:", res["result"])

