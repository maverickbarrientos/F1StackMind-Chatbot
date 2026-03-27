from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os, pandas

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDINGS_MODEL = "models/gemini-embedding-001"

CHROMA_PATH = "./f1stackmind_db"
DATA_PATH = "f1stackmind_dataset.csv"

class Vector():
    
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL) 
        self.vector_store = self._load_vector_store()    
        
    def _load_vector_store(self):
        vector_store = Chroma(
            collection_name="f1stackmind_dataset",
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )
        return vector_store
    
    def build_vectors(self):
        if self.vector_store._collection.count() > 0:
            return
        
        data = pandas.read_csv(DATA_PATH)
        
        documents = []
        ids = []
        
        for i, row in data.iterrows():
            document = Document(
                page_content=f"{row['question']} {row['answer']}",
                id=str(i)
            )
            
            ids.append(str(i))
            documents.append(document)
                
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids  = ids[i:i + batch_size]
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"✓ Embedded {min(i + batch_size, len(documents))}/{len(documents)} documents")
            
            if i + batch_size < len(documents):
                import time
                time.sleep(62) 
        
        print(f"✓ Built vector store with {len(documents)} documents")
        
    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
    