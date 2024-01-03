# A ChatBot for AWS Blogs 🤖 📋

A chatbot to talk to AWS Blogs using local LLMs and semantic search.

## Features

- 🌐 **Multi-Source Blog Indexing**: 
  - Scrapes AWS blog posts from multiple RSS feeds
  - Supports Machine Learning, Security, Big Data, Containers, Databases, Serverless, and Cloud Operations blogs

- 🤖 **Flexible LLM Backend**:
  - Uses [Ollama](https://ollama.com/) for local AI model inference
  - Supports dynamic selection of embedding and text generation models
  - Currently supports Ollama as the primary model backend

- 💾 **Efficient Data Storage**:
  - **Vector Store**: [LanceDB](https://lancedb.com/) for semantic search and embeddings
  - **Metadata Storage**: [DuckDB](https://duckdb.org/) for tracking blog post metadata
  - Semantic chunking of blog posts for improved retrieval

- 🔍 **Advanced Search Capabilities**:
  - Semantic search across blog posts
  - Multiple search types: vector, full-text, hybrid, and re-ranking

## Architecture

### Components

1. **`BlogBuddy.py`**: 
   - Main Streamlit application
   - Handles user interface and model configuration
   - Manages RSS feed refresh and vector store population

2. **`lancedb_utils.py`**:
   - Manages LanceDB vector store
   - Handles semantic chunking of documents
   - Provides advanced search capabilities

3. **`ollama_utils.py`**:
   - Manages Ollama model interactions
   - Retrieves embeddings and generates text responses
   - Dynamically lists available Ollama models

4. **`blog_utils.py`**:
   - Scrapes AWS blog RSS feeds
   - Processes and cleans blog post content
   - Manages DuckDB database for blog post tracking

## Usage

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/)
- [UV package manager](https://astral.sh/uv)

### Installation

#### Install uv

```shell
pip install uv
```

#### Install ollama

Refer to [Ollama Installation quickstart](https://github.com/ollama/ollama/blob/main/README.md#quickstart)


```shell
git clone https://github.com/praveenc/ollama-blog-buddy
```

```shell
cd ollama-blog-buddy
uv venv
uv sync
```

### Launch Streamlit App

```shell
uv run streamlit run app/BlogBuddy.py
```

Once the app is launched:

1. Select an Embedding Model
1. Choose a Text Generation Model (LLM)
1. Click Save
1. Click `Save and Refresh` button to download RSS feeds and populate the vector store
1. Navigate to `Chat` page
1. Start chatting with your AWS blog posts!

#### VectorDB and Storage
We use LanceDB as our vector store. Vectors and metadata are stored locally.

To avoid scraping RSS feeds multiple times, we cache scraped html data to disk and log the scraping activity to DuckDB locally.

### AWS Blogs RSS Feeds

Blog posts are indexed from the below AWS RSS feeds.

- [AWS Machine Learning blogs](https://aws.amazon.com/blogs/machine-learning/feed/)
- [AWS Security blogs](https://aws.amazon.com/blogs/security/feed)
- [AWS Analytics/Big-Data blogs](https://aws.amazon.com/blogs/big-data/feed/)
- [AWS Containers blogs](https://aws.amazon.com/blogs/containers/feed/)
- [AWS Database blogs](https://aws.amazon.com/blogs/databases/feed/)
- [AWS Serverless blogs](https://aws.amazon.com/blogs/compute/tag/serverless/feed/)
- [AWS CloudOperations and Migrations blogs](https://aws.amazon.com/blogs/mt/feed/)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Add your license information here]