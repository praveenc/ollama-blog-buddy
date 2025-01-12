import inspect
import re
import os

from typing import List
import pandas as pd
from utils.blog_utils import BlogsDuckDB
from pathlib import Path
from tavily import TavilyClient
from loguru import logger
import streamlit as st


def tavily_websearch(query: str, num_results: int = 3) -> List[str]:
    """
    Executes a web search query using the Tavily API.

    Args:
        query (str): The search query.
        num_results (int, optional): The maximum number of search results to retrieve. Defaults to 3.

    Returns:
        List[str]: A list of search results.
    """
    api_key = st.secrets.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "Tavily API key not found in secrets. Please configure it in the Settings page."
        )

    logger.info("Executing a web search query using the Tavily API...")
    tavily_client = TavilyClient(api_key=api_key)
    # Step 2. Executing a context search query
    context = tavily_client.get_search_context(query=query, max_results=num_results)
    # Step 2. Executing a Q&A search query
    # answer = tavily_client.qna_search(query=query, max_results=num_results)
    # logger.info(context)
    return context


def generate_function_description(func):
    func_name = func.__name__
    docstring = func.__doc__

    # Get function signature
    sig = inspect.signature(func)
    params = sig.parameters

    # Create the properties for parameters
    properties = {}
    required = []

    # Process the docstring to extract argument descriptions
    arg_descriptions = {}
    if docstring:
        # remove leading/trailing whitespace or leading empty lines and split into lines
        docstring = re.sub(r"^\s*|\s*$", "", docstring, flags=re.MULTILINE)
        lines = docstring.split("\n")
        current_arg = None
        for line in lines:
            line = line.strip()
            if line:
                if ":" in line:
                    # strip leading/trailing whitespace and split into two parts
                    line = re.sub(r"^\s*|\s*$", "", line)
                    parts = line.split(":", 1)
                    if parts[0] in params:
                        current_arg = parts[0]
                        arg_descriptions[current_arg] = parts[1].strip()
                elif current_arg:
                    arg_descriptions[current_arg] += " " + line.strip()

    for param_name, param in params.items():
        param_type = "string"  # Default type; adjust as needed based on annotations
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation.__name__.lower()

        param_description = arg_descriptions.get(
            param_name, f"The name of the {param_name}"
        )

        properties[param_name] = {
            "type": param_type,
            "description": param_description,
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Create the JSON object
    function_description = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring.split("\n")[0]
            if docstring
            else f"Function {func_name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return function_description


def fetch_latest_posts(category: str, days: int = 15) -> List[str]:
    """
    Fetches the latest posts from the BlogsDuckDB database based
    on the specified category and number of days.

    Args:
        category (str): The category of posts to fetch. Valid values are "machinelearning", "security".
        days (int, optional): The number of days to fetch posts for. Defaults to 15.

    Returns:
    list(str): A list of strings representing the latest posts in the specified category and time period.

    """

    def format_to_markdown(posts: List[dict]) -> str:
        if not posts:
            return "No blog posts found for the given category and time period."

        # Markdown table header
        markdown_table = "| Blog URL | Published Date | Summary |\n"
        markdown_table += "|---|---|---|\n"

        # Add rows to the table
        for post in posts:
            markdown_table += (
                f"| [{post['blogpost_url']}]({post['blogpost_url']}) "
                f"| {post['published_date']} "
                f"| {post['blog_summary']} |\n"
            )

        return markdown_table

    _ = BlogsDuckDB(db_dir=Path("../duckdb"))
    conn = _.get_connection()
    SQL = f"""SELECT blogpost_url, published_date, blog_summary FROM blogposts
        WHERE blog_category = '{category}'
        AND published_date > CURRENT_TIMESTAMP - INTERVAL '{days} days'"""
    result = conn.execute(SQL).fetchdf()
    if len(result.values) >= 1:
        results_df = result.to_dict(orient="records")
        results = format_to_markdown(results_df)
    else:
        results = pd.DataFrame().to_dict_()
    return results


def fetch_post_summary(post_url: str) -> str:
    """
    Fetches the summary of a blog post from the BlogsDuckDB database.

    Args:
        post_url (str): The URL of the blog post to fetch the summary for.

    Returns:
    str: A string representing the summary of the blog post.

    """
    conn = BlogsDuckDB(db_dir=Path("../duckdb"))
    results_df = conn.get_post_summary(post_url=str(post_url))
    return results_df


tavily_websearch_fn_schema = {
    "type": "function",
    "function": {
        "name": "tavily_websearch",
        "description": "Executes a web search query using the Tavily API",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for",
                },
                "num_results": {
                    "type": "integer",
                    "description": "The maximum number of search results to retrieve",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

fetch_latest_posts_fn_schema = {
    "type": "function",
    "function": {
        "name": "fetch_latest_posts",
        "description": "Fetches the latest posts from the BlogsDuckDB database",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["machinelearning", "security"],
                    "description": "The category of posts to fetch",
                },
                "days": {
                    "type": "integer",
                    "description": "The number of days to fetch posts for",
                    "default": 15,
                },
            },
            "required": ["category"],
        },
    },
}

available_tools = {
    "fetch_latest_posts": fetch_latest_posts_fn_schema,
    "tavily_websearch": tavily_websearch_fn_schema,
}

# print(generate_function_description(tavily_websearch))
# print(available_tools)
# print(available_tools.get('fetch_latest_posts'))
# print(list(available_tools.values()))
