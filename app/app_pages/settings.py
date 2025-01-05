import streamlit as st
from loguru import logger
from pathlib import Path
import ollama
from datetime import datetime

# function to store widget state
def store_widget_state(key):
    """Store widget state to session state."""
    st.session_state[key] = st.session_state["_" + key]
    logger.debug(f"Selected {key}: {st.session_state[key]}")


# Main title with icon
st.title("⚙️ Application Settings")
st.markdown("Configure your chat application settings here.")

# Model Settings Section
with st.expander("🤖 Model Configuration", expanded=True):
    st.markdown("### Model Backend")
    model_backend = st.radio(
        "Select Model Backend",
        ["Ollama", "OpenAI", "Groq"],
        key="_model_backend",
        help="Select the model backend to be used for chat application.",
        horizontal=True,
        on_change=store_widget_state,
        args=["model_backend"],
        captions=["Local inference", "Cloud API", "Cloud API"]
    )

    st.markdown("---")

    # Model Selection
    if model_backend == "Ollama":
        ollama_models = [model.model for model in ollama.list()["models"]]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔤 Embedding Model")
            embedding_model_name = st.selectbox(
                "Select embedding model",
                options=ollama_models,
                key="_embedding_model_name",
                help="Model used for text embeddings and semantic search",
                on_change=store_widget_state,
                args=["embedding_model_name"],
            )
            
            # Show current selection
            if 'embedding_model_name' in st.session_state:
                st.caption(f"Currently selected: `{st.session_state.embedding_model_name}`")

        with col2:
            st.markdown("### 💭 Chat Model")
            llm_model_name = st.selectbox(
                "Select Text generation model",
                options=ollama_models,
                key="_llm_model_name",
                help="Model used for text generation in chat",
                on_change=store_widget_state,
                args=["llm_model_name"],
            )
            
            # Show current selection
            if 'llm_model_name' in st.session_state:
                st.caption(f"Currently selected: `{st.session_state.llm_model_name}`")
    else:
        st.error("🚧 Other backends are not implemented yet. Please select Ollama as model backend.")

# Storage Settings Section
with st.expander("💾 Storage Configuration", expanded=True):
    st.markdown("### Database Paths")
    
    # DuckDB Configuration
    duckdb_path = Path("./duckdb").absolute()
    st.markdown("#### DuckDB Path")
    st.text_input(
        "Enter DuckDB path",
        value=duckdb_path,
        key="_duckdb_path",
        help="Path to store the DuckDB database for blog metadata",
        on_change=store_widget_state,
        args=["duckdb_path"],
    )
    
    # LanceDB Configuration
    vectorstore_path = Path("./lancedb").absolute()
    st.markdown("#### VectorStore Path")
    st.text_input(
        "Enter vector store path",
        value=vectorstore_path,
        key="_vectorstore_path",
        help="Path to store the LanceDB vector database for embeddings",
        on_change=store_widget_state,
        args=["vectorstore_path"],
    )

# Save Settings
save_col1, save_col2 = st.columns([1, 2])
with save_col1:
    st.markdown("""
    Click **Save Settings** to apply your changes and start using the chat application.
    """)

with save_col2:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        # Create success message placeholder
        success_placeholder = st.empty()
        
        # Show success message with spinner
        with st.spinner("Saving settings..."):
            # Log settings
            logger.debug(f"Selected llm Model: {st.session_state.llm_model_name}")
            logger.debug(f"Selected embedding Model: {st.session_state.embedding_model_name}")
            logger.debug(f"Selected model backend: {st.session_state._model_backend}")
            
            # Show success messages
            st.toast("Settings saved successfully!", icon="✅")
            success_placeholder.success("✅ Settings saved successfully!")
        
        # Show navigation hint
        st.info("👈 Navigate to Chat from the sidebar to start chatting")
        

# Add footer
st.markdown("---")
st.caption("Last updated: " + datetime.now().strftime("%d-%b-%Y %H:%M:%S"))