from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma
db = Chroma(persist_directory="./storage", embedding_function=embeddings)
retriever = db.as_retriever()

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=500)

from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "ailia SDKとは何ですか。"
answer = qa.run(query)

print("Q:", query)
print("A:", answer)