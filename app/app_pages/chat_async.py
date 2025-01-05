import streamlit as st
import asyncio
from ollama import AsyncClient
from loguru import logger

# Custom styling for avatars using Data URLs
ASSISTANT_AVATAR = "https://api.dicebear.com/9.x/bottts/svg?seed=Brooklynn"
# ASSISTANT_AVATAR = "https://api.dicebear.com/7.x/bottts/svg?seed=Lucy"
USER_AVATAR = "https://api.dicebear.com/9.x/lorelei/svg?seed=Mason"
# USER_AVATAR = "https://api.dicebear.com/7.x/avataaars/svg?seed=Felix"

async def generate_response(prompt: str, model_name: str, stream: bool = True) -> str:
    """Generate streaming response using Ollama's AsyncClient"""
    client = AsyncClient()
    messages = [{"role": "user", "content": prompt}]
    
    if stream:
        response_chunks = []
        async for chunk in await client.chat(
            model=model_name,
            messages=messages,
            stream=True
        ):
            response_chunks.append(chunk['message']['content'])
            yield ''.join(response_chunks)
    else:
        response = await client.chat(
            model=model_name,
            messages=messages,
            stream=False
        )
        yield response['message']['content']

async def process_stream(prompt: str, model_name: str, response_placeholder):
    """Process the streaming response"""
    full_response = ""
    async for response_chunk in generate_response(prompt, model_name):
        full_response = response_chunk
        response_placeholder.markdown(full_response + "▌")
    response_placeholder.markdown(full_response)
    return full_response

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Access selected models from session state
llm_model_name = st.session_state.get("llm_model_name", "llama2:latest")

# Display chat header
st.header("💬 Chat with AI")

# Display existing messages
for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Display welcome message if no messages exist
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("👋 Hello! I'm here to help. What would you like to discuss?")

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