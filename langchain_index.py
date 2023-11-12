#!/usr/bin/env python

# usage ./langchain_index.py "Markov-Functional_Interest_Rate_Models.pdf"
import pysqlite_load
from pathlib import Path
from langchain.document_loaders import PDFMinerLoader
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from itertools import chain
import re

def fix_eol(text:str):
    return text.replace("-\n", "")

REG_PAGE_PAT = re.compile(r"\{[\d ]+\{$")
def fix_page(text:str):
    return re.sub(REG_PAGE_PAT, "", text)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".\n")

def get_pdf(path_to_pdf: str):
    loader = PDFMinerLoader(path_to_pdf)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    embeddings = OpenAIEmbeddings()
    db: Chroma = Chroma.from_documents(texts, embeddings, persist_directory="./storage")
    return db

def get_texts(path_to_pdf: str):
    loader = PdfReader(path_to_pdf)
    texts = []
    for page in loader.pages:
        page.extract_text()
        raw_text = page.extract_text()
        raw_text = fix_eol(raw_text)
        raw_text = fix_page(raw_text)
        texts.append(raw_text)
    chunked_texts = text_splitter.split_text("\n".join(texts))
    embeddings = OpenAIEmbeddings()
    db: Chroma = Chroma.from_texts(chunked_texts, embeddings, persist_directory="./storage")
    return db

path_to_pdf = sys.argv[1] if len(sys.argv) > 1 else "Markov-Functional_Interest_Rate_Models.pdf"

# texts = get_pdf(path_to_pdf)
db = get_texts(path_to_pdf)
