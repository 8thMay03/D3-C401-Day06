"""
Streamlit chatbot UI for XanhSM policy RAG system.

Usage:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.chain import build_rag_chain

st.set_page_config(
    page_title="XanhSM Policy Assistant",
    page_icon="🚗",
    layout="centered",
)

st.title("XanhSM - Trợ lý Chính sách Tài xế")
st.caption("Hệ thống truy xuất thông tin chính sách dành cho tài xế XanhSM")


@st.cache_resource
def get_chain():
    return build_rag_chain()


rag_chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Hỏi về chính sách XanhSM..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            response = rag_chain.invoke(question)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
