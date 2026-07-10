# rag_agent_app/frontend/app.py

import streamlit as st
from config import FRONTEND_CONFIG
from session_manager import init_session_state
from ui_components import (
    display_header,
    render_document_upload_section,
    render_agent_settings_section,
    display_chat_history,
    display_trace_events
)
from backend_api import chat_with_backend_agent
import requests, json


# set_page_config must be called ONCE, before any other st. call, and only here.
st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }

            /* Fix: hr must not keep default box border on all sides */
            hr {
                border: none !important;
                border-top: 1px solid rgba(120,120,120,0.25) !important;
                margin: 1rem 0 !important;
            }

            /* Chat message bubbles */
            div[data-testid="stChatMessage"] {
                border-radius: 12px;
                padding: 0.6rem 1rem;
                margin-bottom: 0.4rem;
            }

            /* Chat input: remove any stray outline/shadow, clean rounded field */
            div[data-testid="stChatInput"] {
                box-shadow: none !important;
            }
            div[data-testid="stChatInput"] textarea {
                border-radius: 12px !important;
                box-shadow: none !important;
            }
            div[data-testid="stChatInput"] button {
                border-radius: 8px !important;
            }

            /* Sidebar containers (settings/upload cards) get a bit more breathing room */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 12px;
                margin-bottom: 0.75rem;
            }
        </style>
    """, unsafe_allow_html=True)


def main():
    load_custom_css()
    init_session_state()

    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    display_header()

    with st.sidebar:
        st.markdown("### Configuration")
        render_agent_settings_section()
        render_document_upload_section(fastapi_base_url)

    st.markdown("## 💬 Chat with the Agent")
    st.caption("Ask questions grounded in your uploaded documents, with optional web search.")

    display_chat_history()

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    agent_response, trace_events = chat_with_backend_agent(
                        fastapi_base_url,
                        st.session_state.session_id,
                        prompt,
                        st.session_state.web_search_enabled
                    )
                    st.markdown(agent_response)
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})

                    with st.expander("🔬 Agent Workflow Trace", expanded=False):
                        display_trace_events(trace_events)

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend. Please make sure it's running.")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Could not connect to the backend."})
                except requests.exceptions.RequestException as e:
                    st.error(f"Request error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                except json.JSONDecodeError:
                    st.error("Received an invalid response from the backend.")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Invalid response from backend."})
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Unexpected Error: {e}"})


if __name__ == "__main__":
    main()