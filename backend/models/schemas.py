from pydantic import BaseModel

#Validation for the user request
class QueryRequest(BaseModel):
    query: str


#Validation for the response from the RAG model
class QueryResponse(BaseModel):
    query : str
    final_query : str
    answer : str
    strategy : str
    top_k : int
    rerank : bool
    confidence : float
    grade :  str
    retrieval_failed : bool
    attempts : list
    sources : dict

