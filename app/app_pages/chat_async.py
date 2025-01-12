import streamlit as st
import asyncio
from ollama import AsyncClient
from loguru import logger
from utils.llm_tools import available_tools, tavily_websearch, fetch_latest_posts
from typing import List, Dict, Any
import inspect

# Custom styling for avatars using Data URLs
ASSISTANT_AVATAR = "https://api.dicebear.com/9.x/bottts/svg?seed=Brooklynn"
# ASSISTANT_AVATAR = "https://api.dicebear.com/7.x/bottts/svg?seed=Lucy"
USER_AVATAR = "https://api.dicebear.com/9.x/lorelei/svg?seed=Mason"
# USER_AVATAR = "https://api.dicebear.com/7.x/avataaars/svg?seed=Felix"


async def generate_response(
    prompt: str,
    model_name: str,
    stream: bool = True,
    tools: Dict[str, Any] = available_tools,
) -> str:
    """Generate streaming response using Ollama's AsyncClient"""
    async_client = AsyncClient()
    messages = [{"role": "user", "content": prompt}]
    chat_params = {"model": model_name, "messages": messages, "stream": True}

    if stream:
        response_buffer = []
        if tools:
            # Convert tools dict to list of function descriptions
            tool_list = list(tools.values())
            chat_params["tools"] = tool_list

        logger.info(f"Sending chat params")
        # logger.info(chat_params)

        async for chunk in await async_client.chat(**chat_params):
            # print(chunk.message.tool_calls)
            if chunk.message and chunk.message.tool_calls:
                for tool_call in chunk.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    # Create a status indicator for tool execution
                    with st.status(
                        f"🔍 Executing {function_name}...", expanded=True
                    ) as status:
                        # status.write(f"Processing request with parameters: {function_args}")

                        # Get the actual function from available_tools
                        if function_name in available_tools:
                            tool_fn = globals()[
                                function_name
                            ]  # Get function from globals
                            try:
                                # Execute tool and get result
                                tool_result = (
                                    await tool_fn(**function_args)
                                    if inspect.iscoroutinefunction(tool_fn)
                                    else tool_fn(**function_args)
                                )
                                # Update status to success
                                status.update(
                                    label=f"✅ {function_name} completed",
                                    state="complete",
                                )
                                status.write("Retrieved information successfully!")

                                # logger.info(f"Tool {function_name} result: {tool_result}")
                                # print(tool_result)
                                # Add tool result to messages
                                messages.append(
                                    {
                                        "role": "tool",
                                        "name": function_name,
                                        "content": str(tool_result),
                                    }
                                )

                                # Get final response with tool output
                                async for final_chunk in await async_client.chat(
                                    model=model_name, messages=messages, stream=True
                                ):
                                    if final_chunk.message.get("content"):
                                        response_buffer.append(
                                            final_chunk.message["content"]
                                        )
                                        yield "".join(response_buffer)
                            except Exception as e:
                                status.update(
                                    label=f"❌ Error in {function_name}", state="error"
                                )
                                status.write(f"Error: {str(e)}")
                                logger.error(
                                    f"Error executing tool {function_name}: {str(e)}"
                                )
                                response_buffer.append(
                                    f"\nError executing tool {function_name}: {str(e)}"
                                )
                                yield "".join(response_buffer)

            elif chunk.message and chunk.message.get("content"):
                response_buffer.append(chunk.message["content"])
                yield "".join(response_buffer)
    else:
        response = await async_client.chat(**chat_params)
        yield response["message"]["content"]


async def generate_response_og(
    prompt: str,
    model_name: str,
    stream: bool = True,
    tools: List[Any] = available_tools,
) -> str:
    """Generate streaming response using Ollama's AsyncClient"""
    async_client = AsyncClient()
    user_message = [{"role": "user", "content": prompt}]
    chat_params = {"model": model_name, "messages": user_message, "stream": True}
    if stream:
        response_buffer = []
        if tools:
            chat_params["tools"] = tools
        logger.info(chat_params)
        async for chunk in await async_client.chat(**chat_params):
            content = chunk["message"]["content"]
            response_buffer.append(content)
            yield "".join(response_buffer)
    else:
        response = await async_client.chat(**chat_params)
        yield response["message"]["content"]


async def process_stream(prompt: str, model_name: str, response_placeholder):
    """Process the streaming response"""
    full_response = ""
    # Show thinking indicator while processing starts
    with st.spinner("🤔 Thinking..."):
        async for response_chunk in generate_response(prompt, model_name):
            full_response = response_chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    return full_response


# Initialize chat history in session state
if "messages" not in st.session_state:
    sys_prompt = """You are a helpful AI assistant tasked with answering user questions.
    You have access to two tools: tavily_websearch and fetch_latest_posts.
    For any user questions about blog posts use fetch_latest_posts tool.
    For any user questions about current events, latest news or things that you do not know about, use the tavily_websearch tool to search the web.
    Follow these guidelines:
    - Default to using your own knowledge first
    - Only use tools when your knowledge might be outdated or insufficient
    - Always explain when and why you're using a tool
    - Provide thoughtful responses that combine tool results with your own analysis
    - Format responses using markdown for better readability
    Remember: Just because you have access to tools doesn't mean you need to use them for every query!"""
    st.session_state.messages = [{"role": "system", "content": sys_prompt}]
    # Add a flag to track if welcome message has been shown
    st.session_state.welcome_shown = False

# Access selected models from session state
llm_model_name = st.session_state.get("llm_model_name", "llama3.2:latest")

# Display chat header
st.header("💬 Chat with AI")

# Display existing messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

# Display welcome message if no user interaction exist
if (
    len(st.session_state.messages) == 1 and not st.session_state.welcome_shown
):  # Only system message exists
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(
            "👋 Hello! I'm here to help. What would you like to discuss? "
            "I have access to the following **tools**: "
        )
        st.markdown(
            """
            - Web search (_using Tavily_)
            - List metadata about synced blogs (_LanceDB vectorstore_)
            """
        )
    st.session_state.welcome_shown = True


# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        response_placeholder = st.empty()
        try:
            # Create event loop and run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Process the stream and get the full response
            full_response = loop.run_until_complete(
                process_stream(prompt, llm_model_name, response_placeholder)
            )
            loop.close()

            # Store the complete response
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.error(f"Error in chat generation: {e}")

# # Add clear chat button
# if st.session_state.messages:
#     if st.button("Clear Chat"):
#         st.session_state.messages = []
#         st.rerun()
