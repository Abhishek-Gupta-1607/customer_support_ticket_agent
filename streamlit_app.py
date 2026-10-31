"""Customer Support – Streamlit frontend.

Communicates exclusively through the FastAPI backend.
Never calls the LLM, vector store, or ticket repository directly.
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Support", page_icon="🎧")

# ── session initialization ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# ── sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Info")
    sid = st.session_state.session_id
    st.caption(f"Session: `{sid[:8]}…`")

    if st.button("🔄 New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ── title ────────────────────────────────────────────────────────────────
st.title("🎧 Customer Support")
st.caption("Ask about shipping, payments, returns, accounts, or report an issue.")

# ── render conversation history ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            sources = msg.get("sources")
            ticket_id = msg.get("ticket_id")
            if sources:
                st.caption("📄 Sources: " + ", ".join(sources))
            if ticket_id:
                st.success(f"🎫 Ticket created: **{ticket_id}**")

# ── chat input ───────────────────────────────────────────────────────────
if prompt := st.chat_input("How can we help?"):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call backend API
    with st.chat_message("assistant"):
        with st.spinner("Processing your request…"):
            try:
                with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)) as client:
                    resp = client.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "session_id": st.session_state.session_id,
                            "message": prompt,
                        },
                    )

                if resp.status_code == 422:
                    detail = resp.json().get("detail", "Invalid input.")
                    st.error(f"⚠️ Validation error: {detail}")
                elif resp.status_code == 503:
                    detail = resp.json().get("detail", "Service unavailable.")
                    st.error(f"🔧 {detail}. Please try again later.")
                elif resp.status_code >= 400:
                    detail = resp.json().get("detail", "An error occurred.")
                    st.error(f"❌ Error: {detail}")
                else:
                    try:
                        data = resp.json()
                    except Exception:
                        st.error("❌ Received an invalid response from the server.")
                    else:
                        response_text = data.get("response", "")
                        sources = data.get("sources", [])
                        ticket_id = data.get("ticket_id")

                        st.write(response_text)
                        if sources:
                            st.caption("📄 Sources: " + ", ".join(sources))
                        if ticket_id:
                            st.success(f"🎫 Ticket created: **{ticket_id}**")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources,
                            "ticket_id": ticket_id,
                        })

            except httpx.ConnectError:
                st.error(
                    "🔌 Unable to connect to the support service. "
                    "Please ensure the backend is running and try again."
                )
            except httpx.TimeoutException:
                st.error(
                    "⏱️ The request timed out. Please try again."
                )
            except Exception:
                st.error(
                    "❌ An unexpected error occurred. Please try again."
                )
