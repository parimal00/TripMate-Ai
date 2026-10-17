from rag import retrieve_from_rag
from langchain_core.tools import tool

@tool
def search_uploaded_document(query: str) -> str:
    """
    Search uploaded trip planning documents for recommendations, places, itineraries, and experiences.
    Use this tool whenever the user asks for travel suggestions, itineraries, or details from uploaded guides
    """
    
    return retrieve_from_rag(
        query=query,
    )