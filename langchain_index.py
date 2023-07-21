from langchain.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("AR02ALA_UM01_21J.pdf")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings, persist_directory="./storage")
db.persist()
