import streamlit as st
import pandas as pd
from utils.blog_utils import ScrapeAWSBlogs, BlogsDuckDB
from utils.lancedb_utils import LanceDBManager
from utils.ollama_utils import get_available_ollama_models
from loguru import logger
from pathlib import Path

logger.add(
    f"logs/{Path(__file__).stem}.log", rotation="1 week", backtrace=True, diagnose=True
)


# Function to add RSS feed refresh to the main app
def add_rss_feed_refresh(lancedb_manager, blog_feeds):
    # Add button to submit data
    all_extracted_docs = []
    for k, v in blog_feeds.items():
        with st.spinner(f"Refreshing feed for {k} category..."):
            scraper = ScrapeAWSBlogs(feed_url=v, blog_category=k)
            extracted_docs = scraper.get_processed_docs()
            all_extracted_docs.extend(extracted_docs)

    total_records_placeholder = st.empty()
    docs_to_add = len(all_extracted_docs)
    logger.info(f"Total docs extracted: {docs_to_add}")

    if docs_to_add >= 1:
        # Chunk each document with chunk_text_and_add (semantic chunking)
        with st.status("Chunking docs to add to LanceDB..."):
            total_records = lancedb_manager.chunk_docs_and_add(docs=all_extracted_docs)

        total_records_placeholder.subheader(total_records)
        st.session_state.total_records = total_records
        st.success(f"{total_records} records added to LanceDB successfully.")
    else:
        st.info("All RSS feeds are up to date! ✅")


# Modify the existing app() function to include RSS feed refresh
def app():
    # Create a directory for the DuckDB database
    duckdb_path = Path("./duckdb").absolute()
    duckdb_path.mkdir(exist_ok=True, parents=True)
    # Initialize LanceDB
    vectorstore_path = Path("./lancedb").absolute()
    st.session_state.duckdb_path = str(duckdb_path)
    st.session_state.vectorstore_path = str(vectorstore_path)

    # Create DuckDB table if not exist
    logger.info(f"Creating DuckDB table: {duckdb_path}")
    _ = BlogsDuckDB(duckdb_path)

    # Streamlit app layout
    with st.container():
        left_column, right_column = st.columns([30, 70])

        with left_column:
            # Options for model backend
            st.subheader("Model Backend")
            # Model backend selection
            model_backend = st.radio(
                "Select Model Backend",
                ["Ollama", "OpenAI", "Groq"],
                key="model_backend",
                help="Select the model backend to be used for chat application.",
                horizontal=True,
            )
        with right_column:
            # Configuration options for the embedding model and LLM (existing code)
            st.subheader("Embedding Model")

            if model_backend == "Ollama":
                embedding_model_name = st.selectbox(
                    "Select embedding model",
                    options=get_available_ollama_models(filter="embed"),
                    key="embedding_model_name",
                    help="Select an embedding model to be used for chat application.",
                )
                if not "embedding_model_name" in st.session_state:
                    st.session_state.embedding_model_name = embedding_model_name

                EMBEDDING_DIMENSIONS = 8192
                st.session_state.embeddings_max_length = EMBEDDING_DIMENSIONS
                logger.debug(
                    f"Selected Embedding Model/Dimensions: {embedding_model_name}, {EMBEDDING_DIMENSIONS}"
                )
            else:
                st.error("Not supported yet")

            # Configuration options for selecting LLM
            st.subheader("LLM")

            if model_backend == "Ollama":
                llm_model_name = st.selectbox(
                    "Select Text generation model",
                    options=get_available_ollama_models(filter="llama"),
                    key="llm_model_name",
                    placeholder="Select a text generation model.",
                    help="Select text generation model to be used for chat application.",
                )
                if "llm_model_name" not in st.session_state:
                    st.session_state.llm_model_name = llm_model_name
                logger.debug(f"Selected llm Model: {st.session_state.llm_model_name}")
            else:
                st.error("Not supported yet")

    st.markdown("---")
    with st.container():
        col1, _ = st.columns(2)
        with col1:
            st.markdown("## Refresh Blog Feeds")
            st.caption("Click Refresh to get latest blog posts from RSS feeds.")
            aws_blog_feeds = {
                "machinelearning": "https://aws.amazon.com/blogs/machine-learning/feed/",
                "bigdata": "https://aws.amazon.com/blogs/big-data/feed/",
            }

            feed_df = pd.DataFrame(
                aws_blog_feeds.items(), columns=["Feed", "URL"], index=None
            )
            st.table(feed_df)
            st.markdown("---")

        logger.info(st.session_state)

    # Save button to store all the selected values
    if st.button("Save & Refresh"):
        # Initialize LanceDBManager
        lancedb_manager = LanceDBManager(
            db_uri=st.session_state.vectorstore_path,
            table_name="blogbuddy",
            embed_model_name=st.session_state.embedding_model_name,
        )

        # Add RSS feed refresh section
        add_rss_feed_refresh(lancedb_manager, blog_feeds=aws_blog_feeds)

        with st.sidebar:
            st.subheader("**Configuration**")
            st.markdown(f"**Embedding Model:**`{embedding_model_name}`")
            st.markdown(f"**LLM:** `{llm_model_name}`")


if __name__ == "__main__":
    st.set_page_config(
        page_title="BlogBuddy",
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.extremelycoolapp.com/help",
            "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": "# BlogBuddy",
        },
    )
    st.title("👋 Welcome to BlogBuddy 🤖")
    st.caption(
        "To start, Select a LLM backend and an Embedding Model and click `Save`."
    )
    app()
