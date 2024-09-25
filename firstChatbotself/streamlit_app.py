import streamlit as st
from genrate_response import chatbot_response

st.title("Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What's up"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response (replace this with your actual logic)
    response = f"Chatbot: {prompt}"
    # Generate a response using the imported function
    response = chatbot_response(prompt)
    
    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


    