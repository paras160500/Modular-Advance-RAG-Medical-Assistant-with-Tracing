#-------------------------------------------------------------------------------
#                               Import Statements
#-------------------------------------------------------------------------------

import os 
from dotenv import load_dotenv
load_dotenv()

#-------------------------------------------------------------------------------
#                                 Logic Statements
#-------------------------------------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR")
EMBEDED_MODEL = os.getenv("EMBEDED_MODEL")
