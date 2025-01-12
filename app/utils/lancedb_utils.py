from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import lancedb
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import Reranker
from loguru import logger


class LanceDBManager:
    def __init__(
        self,
        db_uri: str,
        table_name: str = "blogbuddy",
        embed_model_name: str = "nomic-embed-text",
        recreate_table: bool = False,
        print_schema=False,
    ):
        """
        Initialize the LanceDB vector store.

        :param db_uri: The URI for the database directory.
        :param table_name: The name of the table in LanceDB.
        :param embed_model_name: The name of the embedding model to use.
        :param recreate_table: If True, recreate the table even if it exists.
        """
        self.db_uri = Path(db_uri)
        self.table_name = table_name
        self.embed_model_name = embed_model_name
        self.VALID_QUERY_TYPES = [
            "vector",
            "fts",
            "hybrid",
            "rerank_vector",
            "rerank_fts",
        ]

        # Ensure the database directory exists
        if not self.db_uri.exists():
            logger.info(f"Creating database directory: {self.db_uri}")
            self.db_uri.mkdir(parents=True)

        # Connect to LanceDB
        self.db = lancedb.connect(self.db_uri)

        # Get the embedding function
        logger.info(f"Getting embedding function: {self.embed_model_name}")
        self.embed_func = (
            get_registry().get("ollama").create(name=self.embed_model_name)
        )

        # Define the Pydantic model for the table schema
        class BlogBuddy(LanceModel):
            text: str = self.embed_func.SourceField()
            vector: Vector(self.embed_func.ndims()) = self.embed_func.VectorField()
            doc_id: str
            source: str
            title: str
            published: str
            summary: str = None
            authors: List[str] = None
            embed_model_name: str

        self.model_class = BlogBuddy

        # Create or open the table
        if recreate_table or self.table_name not in self.db.table_names():
            if self.table_name in self.db.table_names():
                logger.debug(f"Deleting existing table: {self.table_name}...")
                self.db.drop_table(self.table_name)
            logger.debug(
                f"Creating table: {self.table_name} with embedding model '{self.embed_model_name}'."
            )
            self.table = self.db.create_table(
                self.table_name,
                schema=self.model_class.to_arrow_schema(),
                mode="overwrite",
            )
            logger.debug("Creating full-text-search index on 'text' column...")
            self.table.create_fts_index("text", replace=True)
        else:
            self.table = self.db.open_table(self.table_name)
            logger.debug(f"Opened existing table: {self.table_name}.")

        logger.debug(f"Table: {self.table_name} has {self.table.count_rows()} Records.")
        if print_schema:
            print("====== Schema =======")
            print(self.table.schema)
            print("=====================")

    def chunk_docs_and_add(
        self,
        docs: List[Document],
        breakpoint_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "interquartile",
        breakpoint_threshold: float = 1.5,
    ) -> int:
        embeddings = OllamaEmbeddings(model=self.embed_model_name)
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_threshold,
        )

        # Add records to vector store
        try:
            split_docs = semantic_splitter.split_documents(documents=docs)
            logger.debug(
                f"{len(docs)} docs chunked into {len(split_docs)} chunks using breakpoint type: {breakpoint_type}"
            )
            texts_to_embed = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]
            # update metadatas with embedding model name
            for metadata in metadatas:
                metadata["embed_model_name"] = self.embed_model_name
            record_count = self.add_records(texts=texts_to_embed, metadatas=metadatas)
        except Exception as e:
            logger.debug("Inside chunk_docs_and_add")
            logger.error(f"Error adding records: {e}")
            raise e
        return record_count

    def chunk_text_and_add(
        self,
        text: str,
        metadata: Dict[str, Any],
        breakpoint_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "interquartile",
        breakpoint_threshold: float = 1.5,
    ) -> int:
        """
        Split text into chunks based on token count with overlap.
        Uses the embedding model's own tokenizer for accurate token counting.

        Args:
            text: Text to split
            metadata: Metadata for the text
            breakpoint_type:
            breakpoint_threshold:

        Returns:
            Count of added records
        """
        doc = Document(metadata=metadata, page_content=text)
        embeddings = OllamaEmbeddings(model=self.embed_model_name)
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_threshold,
        )
        logger.info(f"Breakpoint Type: {breakpoint_type}")
        split_docs = semantic_splitter.split_documents(documents=[doc])

        texts_to_embed = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        # Add records to vector store
        try:
            record_count = self.add_records(texts=texts_to_embed, metadatas=metadatas)
        except Exception as e:
            logger.error(f"Error adding records: {e}")
            raise e
        return record_count

    def add_records(
        self, texts: List[str], metadatas: List[Dict[str, Any]], mode: str = "append"
    ):
        """
        Add records to the LanceDB table.

        :param texts: List of text strings to embed and store.
        :param metadatas: List of metadata dictionaries corresponding to the texts.
        :param mode: Mode to add records ('append', 'overwrite', etc.).
        """
        if len(texts) != len(metadatas):
            raise ValueError("The number of texts must match the number of metadatas.")

        records = []
        for text, metadata in zip(texts, metadatas):
            record = {"text": text, **metadata}
            records.append(record)

        logger.info(
            f"Adding {len(records)} records to table: {self.table_name} in mode '{mode}'..."
        )
        self.table.add(records, mode=mode)
        logger.info(f"New record count: {self.table.count_rows()}")
        return int(self.table.count_rows())

    def search(
        self,
        query_string: str,
        query_type: Literal[
            "vector", "fts", "hybrid", "rerank_vector", "rerank_fts"
        ] = "vector",
        top_k: int = 5,
        overfetch_factor: int = 2,
        reranker: Optional[Reranker] = None,
    ) -> pd.DataFrame:
        """
        Search the LanceDB table.

        :param query_string: The query string.
        :param query_type: The type of query ('vector', 'fts', 'hybrid', 'rerank_vector', 'rerank_fts').
        :param top_k: The number of top results to return.
        :param overfetch_factor: Factor to over-fetch results for reranking.
        :param reranker: A Reranker object if using rerank queries.
        :return: Pandas DataFrame containing the query results.
        """
        if query_type not in self.VALID_QUERY_TYPES:
            raise ValueError(
                f"Invalid query type: {query_type}. Valid types: {self.VALID_QUERY_TYPES}"
            )

        if query_type in ["hybrid", "rerank_vector", "rerank_fts"] and reranker is None:
            raise ValueError(f"Reranker must be provided for query type: {query_type}")

        if query_type in ["vector", "fts"]:
            results = (
                self.table.search(query_string, query_type=query_type)
                .limit(top_k)
                .to_pandas()
            )
        elif query_type in ["rerank_vector", "rerank_fts"]:
            results = (
                self.table.search(query_string, query_type=query_type)
                .limit(overfetch_factor * top_k)
                .rerank(reranker=reranker)
                .to_pandas()
            ).head(top_k)
        elif query_type == "hybrid":
            results = (
                self.table.search(query_string, query_type=query_type)
                .limit(overfetch_factor * top_k)
                .rerank(reranker=reranker)
                .to_pandas()
            ).head(top_k)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")

        return results

    def get_total_records(self):
        """
        Get the total number of records in the LanceDB table.
        """
        table = self.db.open_table(self.table_name)
        records = table.search().limit(10000).to_list()
        logger.debug(f"Total records in {self.table_name} = {len(records)}")
        return len(records)
