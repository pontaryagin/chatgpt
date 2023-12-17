#!/usr/bin/env python3
from typing import AsyncIterator, Iterable
import pysqlite_load
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableGenerator
from fastapi import FastAPI
from langserve import add_routes
import asyncio
import uvicorn
from operator import itemgetter

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



template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer should be in Japanese.
"""
prompt = ChatPromptTemplate.from_template(template, verbose=True)

model = ChatOpenAI()

def ajoin_docs(docs: list[Document]):
	ret =  "\n\n".join([doc.page_content for doc in docs])
	# print(f"joint: {ret}\n")
	return ret

async def aformat_answer(answer: AsyncIterator[str]):
	yield "A: "
	async for item in answer:
		yield item

from langchain.chains import ConversationChain

chain = (
	{"context": retriever | RunnableLambda(ajoin_docs),  "question": RunnablePassthrough()}
	| prompt
	| model
	| StrOutputParser()
	# | RunnableGenerator(aformat_answer)
)


app = FastAPI(
  title="My PDF Search API",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

add_routes(
    app,
    chain,
    path="/my_chain",
)

async def main():
	question = "please summarize the document"
	while True:
		print("Q: " + (question or ""), end="")
		query = question or input()
		question = None
		async for chunk in chain.astream(query):
			print(chunk, end="", flush=True)
		print("\n")

# if __name__ == "__main__":
# 	# asyncio.run(main())
#     uvicorn.run(app, host="localhost", port=8000)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    # model = ChatOpenAI(streaming=True)
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
    #         ),
    #         ("human", "{question}"),
    #     ]
    # )
    runnable = chain
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable: Runnable = cl.user_session.get("runnable")  # type: ignore

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()



