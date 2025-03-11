import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import weaviate
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.text_splitter import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
)
from weaviate.classes.query import MetadataQuery

from settings import settings


class FAISSManager:
    def __init__(
        self,
        index_name: str = "faiss_index",
        embedding_model: Optional[Embeddings] = None,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
    ):
        self.index_name = FAISSManager.get_index_path(index_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.embedding_model = embedding_model or OpenAIEmbeddings()

        try:
            self.vectorstore = FAISS.load_local(
                self.index_name,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print("ERROR:", e)
            self.vectorstore = FAISS.from_documents(
                [Document(page_content="")],
                self.embedding_model,
            )
            self.vectorstore.save_local(self.index_name)

        # Mapping of file extensions to their respective loaders
        self.loader_mapping = {
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".pdf": UnstructuredPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
        }

    def add_docs(
        self, file_paths: Union[str, List[str]], metadata: Optional[Dict] = None
    ) -> None:
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        documents = []

        for file_path in file_paths:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_extension = path.suffix.lower()

            # TODO: File extension based text splitting
            if file_extension not in self.loader_mapping:
                raise ValueError(
                    f"Unsupported file type: {file_extension}"
                    f"Supported types are: {', '.join(self.loader_mapping.keys())}"
                )

            loader_class = self.loader_mapping[file_extension]

            if file_extension == ".csv":
                loader = loader_class(file_path, source_column="content")
            else:
                loader = loader_class(file_path)

            loaded_docs = loader.load()

            if metadata:
                for doc in loaded_docs:
                    doc.metadata.update(metadata)

            # Split documents into chunks
            for doc in loaded_docs:
                chunks = self.text_splitter.split_text(doc.page_content)
                documents.extend(
                    [
                        Document(page_content=chunk, metadata=doc.metadata)
                        for chunk in chunks
                    ]
                )

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        else:
            self.vectorstore.add_documents(documents)

        self.vectorstore.save_local(self.index_name)

    def search(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized with documents")

        documents = self.vectorstore.similarity_search(query, k=k, filter=filter)
        return documents

    def get_retriever(self, num_docs: int = 2) -> BaseRetriever:
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized with documents")

        return self.vectorstore.as_retriever(search_kwargs={"k": num_docs})

    @staticmethod
    def get_index_path(index_name: str) -> str:
        # If full path is not provided root path is prepended
        if not ("/" in index_name or "\\" in index_name):
            index_name = f"{settings.ROOT_PATH}/{index_name}"
        return index_name


def valid_or_default(value, default):
    return (
        value
        if value is not None and not (isinstance(value, float) and math.isnan(value))
        else default
    )


class WeaviateManager:
    def __init__(self):
        self.client = weaviate.connect_to_local(port=8081)

    def create_index(self, collection_name: str):
        try:
            # Clear up the schema, so that we can recreate it
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)

            self.client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_huggingface(),
                properties=[
                    Property(
                        name="product_id",
                        data_type=DataType.INT,
                    ),
                    Property(
                        name="caption",
                        data_type=DataType.TEXT,
                    ),
                ],
            )

            products = self.client.collections.get(collection_name)
            products_config = products.config.get()
            print(products_config)

        except Exception as e:
            print(f"Class creation error: {str(e)}")

    def delete_index(self, collection_name: str):
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)

    def close(self):
        self.client.close()

    def add_data(self, collection_name: str, df: pd.DataFrame):
        collection = self.client.collections.get(collection_name)
        with collection.batch.dynamic() as batch:
            for _, row in df.iterrows():
                data_obj = {
                    "product_id": valid_or_default(row.get("id"), 0),
                    "caption": valid_or_default(row.get("caption"), "No Caption"),
                }
                print(data_obj)
                batch.add_object(
                    properties=data_obj,
                )

    def search(self, collection_name: str, question: str):
        collection = self.client.collections.get(collection_name)
        results = collection.query.hybrid(
            query=question,
            query_properties=["caption"],
            max_vector_distance=0.15,
            # vector=HybridVector.near_text(
            #     query="large animal",
            #     move_away=Move(force=0.5, concepts=["mammal", "terrestrial"]),
            # ),
            alpha=0.75,
            return_metadata=MetadataQuery(distance=True),
            # return_metadata=MetadataQuery(score=True, explain_score=True),
            # fusion_type=HybridFusion.RANKED,
            # limit=3,
            # auto_limit=5,
            # filters=filters,
            # group_by=group_by,
        )
        return results


def get_text_splitter(file_extension: str) -> TextSplitter:
    """
    Get the appropriate text splitter based on file extension.

    Args:
        file_extension: The file extension (e.g., 'py', 'md', 'txt')

    Returns:
        TextSplitter: The appropriate text splitter for the file type
    """
    # Default chunk size and overlap
    chunk_size = 1000
    chunk_overlap = 200

    if file_extension == "py":
        # For Python files, use language-specific splitter
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif file_extension == "md":
        # For Markdown, use special separators
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif file_extension in ["js", "ts"]:
        # For JavaScript/TypeScript files
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        # Default text splitter for other file types
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )


def process_document(file_path: str) -> List[Document]:
    """Process a document and split it into chunks."""
    # Get file extension
    file_extension = file_path.split(".")[-1].lower() if "." in file_path else ""

    # Get appropriate text splitter
    text_splitter = get_text_splitter(file_extension)

    # Load and split the document
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create metadata
    metadata = {"source": file_path}

    # Split text into chunks
    docs = text_splitter.create_documents(texts=[text], metadatas=[metadata])

    return docs
