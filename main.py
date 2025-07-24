import streamlit as st
from io import StringIO
import asyncio
import shutil
import os

VECTOR_STORE_PATH = "chroma_db"

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from lib.ai import upload_file, generate_response


def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"name": "ai", "value": "How can I help you today?"}
        ]

    if uploaded_files := st.file_uploader("Choose a file", accept_multiple_files=True):
        for uploaded_file in uploaded_files:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            upload_file(string_data)

    for message in st.session_state["messages"]:
        with st.chat_message(name=message["name"]):
            st.write(message["value"])

    if prompt := st.chat_input("Say something"):
        st.session_state["messages"].append({"name": "user", "value": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in generate_response(prompt):
                print(chunk)
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        st.session_state["messages"].append({"name": "ai", "value": full_response})


if __name__ == "__main__":
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    main()
