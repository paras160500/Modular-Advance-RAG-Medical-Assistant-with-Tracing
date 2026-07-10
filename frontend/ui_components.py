# rag_agent_app/frontend/ui_components.py

import streamlit as st
from backend_api import upload_document_to_backend, chat_with_backend_agent
from session_manager import init_session_state


def display_header():
    """Renders the main title and introductory markdown."""
    st.title("🤖 AI Agent Chatbot")
    st.markdown("Ask me anything! I can answer questions using my internal knowledge (RAG) or by searching the web.")
    st.divider()


def render_document_upload_section(fastapi_base_url: str):
    """Renders the UI for uploading PDF documents to the knowledge base."""
    with st.container(border=True):
        st.markdown("**📄 Upload Document to Knowledge Base**")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type="pdf", key="pdf_uploader",
            label_visibility="collapsed"
        )

        if st.button("Upload PDF", key="upload_pdf_button", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        upload_data = upload_document_to_backend(fastapi_base_url, uploaded_file)
                        st.success(
                            f"'{upload_data.get('filename')}' uploaded — "
                            f"{upload_data.get('processed_chunks')} pages processed."
                        )
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
            else:
                st.warning("Please choose a PDF file first.")


def render_agent_settings_section():
    """Renders the section for agent settings, including the web search toggle."""
    with st.container(border=True):
        st.markdown("**⚙️ Agent Settings**")
        st.session_state.web_search_enabled = st.toggle(
            "Enable Web Search 🌐",
            value=st.session_state.web_search_enabled,
            help="If enabled, the agent can search the web when its knowledge base is insufficient."
        )


def display_chat_history():
    """Displays all messages currently in the session state chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_trace_events(trace_events: list):
    """Renders the detailed agent workflow trace in an expandable section."""
    if not trace_events:
        return

    icon_map = {
        'router': "➡️",
        'rag_lookup': "📚",
        'web_search': "🌐",
        'answer': "💡",
        '__end__': "✅"
    }

    for event in trace_events:
        icon = icon_map.get(event['node_name'], "⚙️")
        st.markdown(f"**{icon} Step {event['step']}: {event['node_name']}**")
        st.caption(event['description'])

        if event['node_name'] == 'rag_lookup' and 'sufficiency_verdict' in event['details']:
            verdict = event['details']['sufficiency_verdict']
            if verdict == "Sufficient":
                st.success(f"RAG Verdict: {verdict} — relevant info found in knowledge base.")
            else:
                st.warning(f"RAG Verdict: {verdict} — no sufficient info found. Diverting to web search.")
            if 'retrieved_content_summary' in event['details']:
                st.markdown(f"**Retrieved summary:** `{event['details']['retrieved_content_summary']}`")

        elif event['node_name'] == 'web_search' and 'retrieved_content_summary' in event['details']:
            st.markdown(f"**Web search summary:** `{event['details']['retrieved_content_summary']}`")

        elif event['node_name'] == 'router' and 'router_override_reason' in event['details']:
            st.info(f"Router override: {event['details']['router_override_reason']}")
            st.json({
                "initial_decision": event['details']['initial_decision'],
                "final_decision": event['details']['final_decision']
            })

        elif event['details']:
            st.json(event['details'])

        st.divider()