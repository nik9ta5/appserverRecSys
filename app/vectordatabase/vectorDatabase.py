from langchain_chroma import Chroma
from langchain_core.documents import Document


def CreateVectorDatabase(collection_name, embedding_function, persist_directory = "./vectordatabase/chromoDB_med"):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
