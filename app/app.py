import streamlit as st


def check_session_variables():
    required_vars = [
        "duckdb_path",
        "vectorstore_path",
        "embedding_model_name",
        "llm_model_name",
        "total_records",
    ]

    return all(var in st.session_state for var in required_vars)


st.set_page_config(page_title="BlogBuddy", page_icon=":material/add_circle:")
settings_page = st.Page(
    "app_pages/settings.py", title="Settings", icon=":material/settings:", default=True
)
data_refresh_page = st.Page(
    "app_pages/data_refresh.py", title="Data Refresh", icon=":material/refresh:"
)
chat_page = st.Page("app_pages/chat_async.py", title="Chat", icon=":material/chat:")

sidebar_menu = {
    "Settings": [settings_page],
    "Data Refresh": [data_refresh_page],
    "Chat": [chat_page],
}

pg = st.navigation(sidebar_menu)

pg.run()
