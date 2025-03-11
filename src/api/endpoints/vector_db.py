import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from pydantic import BaseModel

from agents.rag import FAISSManager, WeaviateManager
from api.dependencies import verify_bearer

router = APIRouter(
    prefix="/vector_db",
    tags=["vector_db"],
    dependencies=[Depends(verify_bearer)],
)


class DocumentsInput(BaseModel):
    index_name: str = "faiss_index"
    file_paths: list[str]
    metadata: dict | None = None


class FaissQuery(BaseModel):
    index_name: str = "faiss_index"
    query: str
    k: int = 4


class DataInput(BaseModel):
    collection_name: str
    data: dict = {}
    metadata: dict | None = None


class WeaviateQuery(BaseModel):
    collection_name: str
    query: str


@router.post("/faiss/index")
async def index_faiss(index_name: str):
    try:
        FAISSManager(index_name=index_name)
        return {"message": "FAISS index created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faiss/add")
async def add_documents(input_data: DocumentsInput):
    try:
        manager = FAISSManager(index_name=input_data.index_name)
        manager.add_docs(
            input_data.file_paths,
            input_data.metadata,
        )
        return {"message": "Documents added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/faiss/search")
async def search_documents(faiss_query: FaissQuery):
    try:
        manager = FAISSManager(index_name=faiss_query.index_name)
        results = manager.search(
            faiss_query.query,
            faiss_query.k,
        )
        return {
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weaviate/index")
async def index_weaviate(collection_name: str):
    try:
        manager = WeaviateManager()
        manager.create_index(collection_name)
        return {"message": "Weaviate index created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weaviate/add")
async def add_to_weaviate(data_input: DataInput):
    try:
        manager = WeaviateManager()
        df = pd.DataFrame(data_input.data)
        manager.add_data(data_input.collection_name, df)
        return {"message": "Data added to Weaviate successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weaviate/search")
async def search_weaviate(weaviate_query: WeaviateQuery):
    try:
        manager = WeaviateManager()
        results = manager.search(
            weaviate_query.collection_name,
            weaviate_query.query,
        )
        return {"results": results.objects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
