import streamlit as st
import pandas as pd
from utils.blog_utils import ScrapeAWSBlogs, BlogsDuckDB
from utils.lancedb_utils import LanceDBManager
from loguru import logger
from pathlib import Path

aws_blog_feeds = {
    "machinelearning": "https://aws.amazon.com/blogs/machine-learning/feed/",
    "bigdata": "https://aws.amazon.com/blogs/big-data/feed/",
    "security": "https://aws.amazon.com/blogs/security/feed"
}

# Create a directory for the DuckDB database
duckdb_path = Path("./duckdb").absolute()
duckdb_path.mkdir(exist_ok=True, parents=True)
# Initialize LanceDB
vectorstore_path = Path("./lancedb").absolute()

st.session_state.duckdb_path = str(duckdb_path)
st.session_state.vectorstore_path = str(vectorstore_path)

# Initialize LanceDBManager
lancedb_manager = LanceDBManager(
    db_uri=str(vectorstore_path),
    table_name="blogbuddy",
    embed_model_name=st.session_state.embedding_model_name,
)


def add_rss_feed_refresh(lancedb_manager, blog_feeds, records_metric):
    all_extracted_docs = []
    progress_bar = st.progress(0)
    feed_count = len(blog_feeds)
    
    for idx, (category, url) in enumerate(blog_feeds.items()):
        progress_text = st.empty()
        progress_text.text(f"Processing {category} feed...")
        
        scraper = ScrapeAWSBlogs(feed_url=url, blog_category=category)
        extracted_docs = scraper.get_processed_docs()
        all_extracted_docs.extend(extracted_docs)
        
        progress_bar.progress((idx + 1) / feed_count)
        progress_text.empty()    
    
    # total_records_placeholder = st.empty()
    docs_to_add = len(all_extracted_docs)
    logger.info(f"Total docs extracted: {docs_to_add}   ")

    if docs_to_add == 0:
        st.toast('All RSS feeds are up to date! 🎉', icon='🎉')
        st.success("No new content to add - database is current! 🎉")
    else:
        # Chunk each document with chunk_text_and_add (semantic chunking)
        with st.status(f"💫 Processing {docs_to_add} new documents...") as status:
            status.update(label=f"Chunking and embedding {docs_to_add} documents...")
            total_records = lancedb_manager.chunk_docs_and_add(docs=all_extracted_docs)
            status.update(label="✅ Processing complete!", state="complete")
        # Update the records metric
            records_metric.metric(
                "Total Records",
                f"{total_records:,}",
                delta=docs_to_add,
                help="Total number of chunks in the vector database"
            )
            
        st.success(f"Successfully added {docs_to_add} new documents to the database!")
        st.session_state.total_records = total_records

        

# Header Section
st.title("🔄 Blog Data Refresh")
st.markdown("Keep your blog database current with the latest AWS content.")

# Dashboard Stats
col1, col2, col3 = st.columns(3)
with col1:
    records_metric = st.empty()
    records_metric.metric(
        "Total Records",
        f"{lancedb_manager.get_total_records():,}",
        help="Total number of chunks in the vector database"
    )
with col2:
    st.metric(
        "RSS Feeds",
        len(aws_blog_feeds),
        help="Number of RSS feeds being monitored"
    )
with col3:
    st.metric(
        "Categories",
        len(set(aws_blog_feeds.keys())),
        help="Number of unique blog categories"
    )

# RSS Feeds Section
st.markdown("## 📊 Monitored RSS Feeds")
with st.expander("View RSS Feed Details", expanded=True):
    feed_df = pd.DataFrame(
        [
            {
                "Category": category.capitalize(),
                "Feed URL": url,
                "Status": "Active ✅"
            }
            for category, url in aws_blog_feeds.items()
        ]
    )
    st.dataframe(
        feed_df,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn(
                "Category",
                help="Blog category",
                width="medium"
            ),
            "Feed URL": st.column_config.LinkColumn(
                "Feed URL",
                help="Click to open RSS feed",
                width="large"
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                help="Current feed status",
                width="small"
            )
        }
    )

# Refresh Section
st.markdown("## 🚀 Update Database")
refresh_col1, refresh_col2 = st.columns([2, 1])
with refresh_col1:
    st.markdown("""
        Click refresh to:
        - Fetch latest blog posts
        - Process new content
        - Update vector database
    """)
with refresh_col2:
    if st.button("Refresh Now", type="primary", use_container_width=True):
        _ = BlogsDuckDB(duckdb_path)
        add_rss_feed_refresh(lancedb_manager, blog_feeds=aws_blog_feeds, records_metric=records_metric)

# Footer
st.markdown("---")
st.caption("Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))