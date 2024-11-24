import shutil
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


class ChromaDB:
    def __init__(self, data_path=r"data/", chroma_path=r"chroma"):
        self.data_path = data_path
        self.chroma_path = chroma_path
        load_dotenv('.env')

    def create_database(self):
        docs = self._load_documents()
        chunks = self._generate_document_chunks(docs)
        self._save_chunks_to_database(chunks=chunks, batch_size=5000)

    def _save_chunks_to_database(self, chunks, batch_size):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            db = Chroma.from_documents(
                documents=batch,
                embedding=OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),
                persist_directory=self.chroma_path
            )

    @staticmethod
    def _generate_document_chunks(docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(docs)
        return chunks

    def _load_documents(self):
        loader = DirectoryLoader(self.data_path, glob="**/*.txt", show_progress=True)
        docs = loader.load()

        return docs

    def load_database(self, embedding_function):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=embedding_function)
        return db
