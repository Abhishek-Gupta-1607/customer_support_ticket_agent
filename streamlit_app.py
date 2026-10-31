"""Customer Support – Streamlit Modern Frontend.

Communicates exclusively through the FastAPI backend REST API.
Never calls the LLM, vector store, or ticket repository directly.
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Modern Clean UI ────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    /* Header hero styling */
    .hero-container {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-title {
        color: #1e1b4b;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #4338ca;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Source tag chips */
    .source-chip {
        display: inline-flex;
        align-items: center;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #334155;
        font-weight: 500;
        margin-right: 6px;
        margin-top: 6px;
    }

    /* Ticket Confirmation Card */
    .ticket-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
        color: #065f46;
        font-size: 0.95rem;
    }
    .ticket-id {
        font-weight: 700;
        color: #047857;
        font-family: monospace;
        font-size: 1.05rem;
    }

    /* Quick prompt button style */
    div[data-testid="stButton"] button {
        border-radius: 20px;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #4f46e5;
        color: #4f46e5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ──────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "prefill_prompt" not in st.session_state:
    st.session_state.prefill_prompt = None


# ── Health Check Helper ───────────────────────────────────────────────────
def check_backend_health() -> bool:
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{API_BASE_URL}/health")
            return resp.status_code == 200 and resp.json().get("status") == "ready"
    except Exception:
        return False


# ── Sidebar Navigation & Tools ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎧 Support Portal")
    
    # Live backend status indicator
    is_ready = check_backend_health()
    if is_ready:
        st.success("🟢 System Operational", icon="✅")
    else:
        st.warning("🟠 Backend Connecting...", icon="⚠️")

    st.markdown("---")
    st.markdown("#### 💬 Active Session")
    sid = st.session_state.session_id
    st.code(sid, language="text")

    if st.button("🔄 New Conversation", use_container_width=True, type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.prefill_prompt = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔍 Ticket Status Lookup")
    with st.expander("Search by Ticket ID", expanded=False):
        lookup_id = st.text_input("Ticket ID", placeholder="CST-2026-0001", key="lookup_input")
        if st.button("Check Status", use_container_width=True):
            if lookup_id.strip():
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.get(f"{API_BASE_URL}/tickets/{lookup_id.strip()}")
                        if resp.status_code == 200:
                            tdata = resp.json()
                            st.success(f"Status: **{tdata.get('status', 'open').upper()}**")
                            st.write(f"**Customer:** {tdata.get('customer_name')}")
                            st.write(f"**Category:** {tdata.get('category')}")
                            st.write(f"**Summary:** {tdata.get('summary')}")
                            st.caption(f"Created: {tdata.get('created_at')}")
                        elif resp.status_code == 404:
                            st.error("Ticket not found.")
                        else:
                            st.error(f"Error {resp.status_code} while looking up ticket.")
                except Exception as e:
                    st.error(f"Lookup error: {e}")
            else:
                st.warning("Please enter a valid ticket ID.")

    st.markdown("---")
    st.caption("Powered by **FastAPI**, **LangGraph**, **ChromaDB**, and **Groq**.")


# ── Main Header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎧 Customer Support AI</div>
        <div class="hero-subtitle">Instant answers to shipping, payment, return policies, or help creating support tickets.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Quick Questions Pill Bar ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("##### 💡 Frequently Asked Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 How long does standard shipping take?", use_container_width=True):
            st.session_state.prefill_prompt = "How long does standard shipping take?"
        if st.button("💳 I have a pending payment. What does it mean?", use_container_width=True):
            st.session_state.prefill_prompt = "I have a pending payment. What does it mean?"
    with col2:
        if st.button("🔄 What is the return eligibility period?", use_container_width=True):
            st.session_state.prefill_prompt = "What is the return eligibility period?"
        if st.button("⚠️ My payment was charged twice. I want to report this.", use_container_width=True):
            st.session_state.prefill_prompt = "My payment was charged twice. I want to report this."
    st.markdown("---")


# ── Message History Rendering ─────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else "🎧"
    with st.chat_message(role, avatar=avatar):
        st.write(msg["content"])
        
        # Render sources if available
        sources = msg.get("sources")
        if sources:
            source_html = " ".join([f'<span class="source-chip">📄 {s}</span>' for s in sources])
            st.markdown(f'<div style="margin-top: 8px;">{source_html}</div>', unsafe_allow_html=True)
            
        # Render ticket card if created
        ticket_id = msg.get("ticket_id")
        if ticket_id:
            st.markdown(
                f"""
                <div class="ticket-card">
                    🎫 <b>Official Support Ticket Created:</b> <span class="ticket-id">{ticket_id}</span><br>
                    <small>Keep this ticket ID for tracking your inquiry with our support team.</small>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Chat Input & Processing ───────────────────────────────────────────────
prompt = st.chat_input("Ask a question or report an issue...")

# If user clicked a quick question pill, use that as prompt
if st.session_state.prefill_prompt and not prompt:
    prompt = st.session_state.prefill_prompt
    st.session_state.prefill_prompt = None

if prompt:
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # 2. Call FastAPI backend
    with st.chat_message("assistant", avatar="🎧"):
        with st.spinner("Analyzing knowledge base & reasoning..."):
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
                    st.error(f"🔧 Service is starting up: {detail}. Please try again shortly.")
                elif resp.status_code == 502:
                    detail = resp.json().get("detail", "Upstream processing failure.")
                    st.error(f"⚠️ {detail}")
                elif resp.status_code >= 400:
                    detail = resp.json().get("detail", "An error occurred.")
                    st.error(f"❌ Error: {detail}")
                else:
                    try:
                        data = resp.json()
                    except Exception:
                        st.error("❌ Received an unreadable response from the server.")
                    else:
                        response_text = data.get("response", "")
                        sources = data.get("sources", [])
                        ticket_id = data.get("ticket_id")

                        st.write(response_text)
                        
                        if sources:
                            source_html = " ".join([f'<span class="source-chip">📄 {s}</span>' for s in sources])
                            st.markdown(f'<div style="margin-top: 8px;">{source_html}</div>', unsafe_allow_html=True)
                            
                        if ticket_id:
                            st.markdown(
                                f"""
                                <div class="ticket-card">
                                    🎫 <b>Official Support Ticket Created:</b> <span class="ticket-id">{ticket_id}</span><br>
                                    <small>Keep this ticket ID for tracking your inquiry with our support team.</small>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources,
                            "ticket_id": ticket_id,
                        })

            except httpx.ConnectError:
                st.error(
                    "🔌 **Cannot connect to the Support API.** "
                    "Please ensure the FastAPI backend is running on `127.0.0.1:8000`."
                )
            except httpx.TimeoutException:
                st.error("⏱️ **The request timed out.** Please try again.")
            except Exception as exc:
                st.error(f"❌ **An unexpected error occurred:** {exc}")
