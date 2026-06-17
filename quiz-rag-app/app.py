# app.py
import streamlit as st
import os
from pathlib import Path

# Import our custom modules
import rag_engine
import quiz_generator
from prompt_styles import STYLE_DISPLAY_NAMES, get_style

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Curated Quiz Wizard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject Custom CSS Aesthetics ──────────────────────────────────────────────
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session State Initialisation ──────────────────────────────────────────────
# We keep track of whether a document is parsed to avoid re-embedding on every rerun
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = ""

# ── Header Section ────────────────────────────────────────────────────────────
st.title("🧠 Curated Prompt Styles: Quiz Wizard")
st.markdown(
    "Upload your notes, pick an interactive prompt engine style, "
    "and instantly test your understanding with custom target variations."
)
st.write("---")

# ── Sidebar Layout: Controls & Inputs ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 1. Groq Key Verification
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("🔑 GROQ_API_KEY missing in `.env` file!")
        st.stop()
    else:
        st.success("⚡ Groq Engine Linked Successfully!")

    st.write("---")
    
    # 2. Document Upload Field
    st.subheader("📁 Document Repository")
    uploaded_file = st.file_uploader(
        "Drop your engineering notes or textbook chapter (PDF)",
        type=["pdf"],
        help="Reads text contents locally via CPU-safe vectors."
    )
    
    # Handle New Document Processing
    if uploaded_file is not None:
        if st.session_state.current_file_name != uploaded_file.name:
            with st.spinner("Parsing text and setting up vector workspace..."):
                try:
                    # Pass the raw file bytes right to the RAG engine
                    file_bytes = uploaded_file.read()
                    num_chunks = rag_engine.ingest_pdf(file_bytes)
                    
                    st.session_state.pdf_processed = True
                    st.session_state.current_file_name = uploaded_file.name
                    st.toast(f"✅ Indexed {num_chunks} text chunks!")
                except Exception as e:
                    st.error(f"Failed to compile PDF: {e}")
                    st.session_state.pdf_processed = False
    else:
        # Reset state if user removes the file
        st.session_state.pdf_processed = False
        st.session_state.current_file_name = ""

    st.write("---")
    
    # 3. Style Engine Select Box
    st.subheader("🎭 Prompt Style Paradigm")
    selected_style_name = st.selectbox(
        "Select Generation Matrix:",
        options=STYLE_DISPLAY_NAMES,
        help="Changes the baseline system instruction architecture of the AI model."
    )
    
    # Dynamically display the subtitle subtitle of the matching blueprint
    current_style_meta = get_style(selected_style_name)
    st.markdown(f"**Persona Logic:** *{current_style_meta.description}*")


# ── Main Stage Area ───────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("💡 **Welcome Netrunner!** To start generating styled questions, drop a reference syllabus or notes PDF into the left sidebar repository.")
else:
    # Build two semantic columns for clean dashboard organization
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown(
            f"""
            <div class='vtu-card'>
                <h4>📄 File Workspace Active</h4>
                <p style='font-size: 0.95rem; color:#888;'>Analyzing parameters for: <br><b>{st.session_state.current_file_name}</b></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.subheader("🎯 Focus Matrix")
        topic_input = st.text_input(
            "What specific topic or core system block do you want to target?",
            placeholder="e.g., Stack Operations, Pointers, Rectifiers",
            help="The RAG core searches the database using this precise contextual phrase."
        )
        
        generate_btn = st.button("✨ Initialize Generation Sequence")

    with col2:
        st.subheader("📝 Live Console Output")
        
        if generate_btn:
            if not topic_input.strip():
                st.warning("⚠️ Enter an internal topic scope constraint before processing.")
            else:
                # Target visual card setup for streaming context cleanly
                output_container = st.empty()
                
                try:
                    # Call our public stream workflow from quiz_generator.py
                    quiz_stream = quiz_generator.generate_quiz(
                        display_name=selected_style_name,
                        topic=topic_input
                    )
                    
                    # Consume the streaming iterator using Streamlit's stream render component
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write_stream(quiz_stream)
                        
                except Exception as e:
                    st.error(f"Execution Error during pipeline process: {e}")