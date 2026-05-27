from backend.models.schemas import QueryRequest,QueryResponse
from rag_pipeline.app import Pipeline
from fastapi import APIRouter,Request

router = APIRouter()


@router.post("/query/",response_model=QueryResponse)
def post_query(body:QueryRequest, request:Request):
    pipe = request.app.state.pipe
    res = pipe.query(body.query)
    return QueryResponse(**res)
