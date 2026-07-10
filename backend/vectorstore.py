#-------------------------------------------------------------------------------
#                               Import Statements
#-------------------------------------------------------------------------------

import os,time
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY,EMBEDED_MODEL,PINECONE_INDEX_NAME,PINECONE_ENVIRONMENT

#-------------------------------------------------------------------------------
#                               Initializing the var
#-------------------------------------------------------------------------------

# Setting up environ variable for pinecone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
# Initialize the pinecone client 
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name = EMBEDED_MODEL)

#-------------------------------------------------------------------------------
#                               Logical Functions
#-------------------------------------------------------------------------------

def get_retriever():
    """
        Initialize the pinecone vectorstore retriever
        Returns:
            Pinecone vectorstore retriever
    """
    # Ensure the index is exists or not
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # If index is not present
        print("Creating the Index...")
        pc.create_index(
            name = PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec = ServerlessSpec(cloud = "aws" , region = PINECONE_ENVIRONMENT)
        )
        print(f"Created pinecone index {PINECONE_INDEX_NAME}")

    # Creating Vector store 
    vectorstore = PineconeVectorStore(index_name = PINECONE_INDEX_NAME , embedding=embeddings)
    # Returning retriever
    return vectorstore.as_retriever() 


def add_document(text_content : str):
    """
        Add a sinle document to the Pinecone vectorstore.
        Splits the text into chunks before embedding and upserting.
        Args:
            text_content : content to push on the pinecone 
    """

    # Ensure the index is exists or not
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # If index is not present
        print("Creating the Index...")
        pc.create_index(
            name = PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec = ServerlessSpec(cloud = "aws" , region = PINECONE_ENVIRONMENT)
        )
        time.sleep(10)
        print(f"Created pinecone index {PINECONE_INDEX_NAME}")



    if not text_content:
        raise ValueError("Please pass document with proper content")
    print(text_content)
    # Split Doc
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True 
    )

    # Create Langchain Document to store the text 
    documents = text_splitter.create_documents([text_content])
    print("Splitting document into chunk for indexing...")

    # get vectorstore instance to add doc to the database 
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME , embedding=embeddings)
    vectorstore.add_documents(documents)
    print("Document added Successfully to Pinecone.")

    