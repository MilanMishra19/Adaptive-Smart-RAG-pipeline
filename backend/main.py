from fastapi import FastAPI
from rag_pipeline.app import Pipeline
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.routes.query import router 


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("RAG pipeline loading")
    app.state.pipe = Pipeline()
    print("RAG loaded..")
    yield
    print("RAG stopped")
app = FastAPI(title="RAG API",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)

@app.get("/")
def root():
    return {"message":"RAG API running successfully"}