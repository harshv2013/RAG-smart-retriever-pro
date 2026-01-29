# import streamlit as st
# import sys
# from pathlib import Path

# # Add src to path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

# from rag_system import rag

# st.set_page_config(
#     page_title="SmartRetriever Pro",
#     layout="wide"
# )

# st.title("🧠 SmartRetriever Pro")

# tab_query, tab_upload = st.tabs(["❓ Query", "📄 Upload Documents"])
# st.caption("Upload documents and ask questions using RAG")

# with tab_query:
#     st.subheader("Ask a question")

#     query = st.text_input("Enter your question")

#     if st.button("Search"):
#         if not query.strip():
#             st.warning("Please enter a question")
#         else:
#             with st.spinner("Thinking..."):
#                 result = rag.query(query)

#             st.markdown("### ✅ Answer")
#             st.write(result["answer"])

#             st.markdown("### 📚 Sources")
#             for src in set(result["sources"]):
#                 st.write(f"- {src}")
                

# with tab_upload:
#     st.subheader("Upload documents")

#     uploaded_files = st.file_uploader(
#         "Upload text files",
#         type=["txt", "pdf"],
#         accept_multiple_files=True
#     )

#     if st.button("Process Documents"):
#         if not uploaded_files:
#             st.warning("Please upload at least one file")
#         else:
#             upload_dir = PROJECT_ROOT / "data" / "documents"
#             upload_dir.mkdir(parents=True, exist_ok=True)

#             for file in uploaded_files:
#                 file_path = upload_dir / file.name
#                 with open(file_path, "wb") as f:
#                     f.write(file.getbuffer())

#             with st.spinner("Processing documents..."):
#                 results = rag.add_documents_from_directory(str(upload_dir))

#             success = sum(1 for r in results if r["success"])
#             st.success(f"Processed {success}/{len(results)} documents")


# #########################################################################

# import sys
# import streamlit as st
# from pathlib import Path
# import tempfile
# import json


# # from pathlib import Path

# # Add src to path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

# # from rag_system import rag
# from src.rag_system import rag

# # ----------------------------
# # Page config
# # ----------------------------
# st.set_page_config(
#     page_title="SmartRetriever Pro",
#     page_icon="🧠",
#     layout="wide"
# )

# st.title("🧠 SmartRetriever Pro")
# st.caption("Production-grade Retrieval Augmented Generation system")

# # =============================
# # Sidebar Controls
# # =============================
# st.sidebar.header("⚙️ Retrieval Settings")

# top_k = st.sidebar.slider(
#     "Top-K Chunks",
#     min_value=1,
#     max_value=10,
#     value=5,
#     step=1,
#     help="Number of chunks retrieved from FAISS"
# )

# use_cache = st.sidebar.checkbox(
#     "Use Cache",
#     value=True
# )

# # ----------------------------
# # Sidebar – System Status
# # ----------------------------
# with st.sidebar:
#     st.header("⚙️ System Status")

#     try:
#         stats = rag.get_stats()

#         st.success("System Online")

#         st.metric("📄 Documents", stats["database"]["total_documents"])
#         st.metric("🧩 Chunks", stats["database"]["total_chunks"])
#         st.metric("🔢 FAISS Vectors", stats["faiss"]["total_vectors"])
#         st.metric("⚡ Cache Keys", stats["cache"].get("total_keys", 0))

#         if st.button("🗑️ Clear Cache"):
#             rag.clear_cache()
#             st.success("Cache cleared")

#     except Exception as e:
#         st.error("System not ready")
#         st.caption(str(e))

# # ----------------------------
# # Tabs
# # ----------------------------
# tab_upload, tab_query, tab_debug = st.tabs(
#     ["📤 Upload Documents", "🔍 Ask Query", "🧪 Debug / Inspect"]
# )

# # ==========================================================
# # 📤 Upload Documents
# # ==========================================================
# with tab_upload:
#     st.subheader("📤 Upload Documents")

#     uploaded_files = st.file_uploader(
#         "Upload text documents",
#         type=["txt", "md"],
#         accept_multiple_files=True
#     )

#     if uploaded_files:
#         if st.button("🚀 Process Documents"):
#             with st.spinner("Processing documents..."):
#                 results = []

#                 for file in uploaded_files:
#                     # Save temp file
#                     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#                         tmp.write(file.read())
#                         tmp_path = tmp.name

#                     try:
#                         result = rag.add_document(
#                             filepath=tmp_path,
#                             filename=file.name
#                         )
#                         results.append(result)

#                     except Exception as e:
#                         results.append({
#                             "success": False,
#                             "filename": file.name,
#                             "error": str(e)
#                         })

#                 st.success("Document processing completed")

#                 st.subheader("📄 Results")
#                 st.json(results)

# # ==========================================================
# # 🔍 Query UI
# # ==========================================================
# with tab_query:
#     st.subheader("🔍 Ask a Question")

#     query = st.text_input(
#         "Enter your question",
#         placeholder="e.g. What is Python?"
#     )

#     col1, col2 = st.columns(2)
#     with col1:
#         use_cache = st.checkbox("Use cache", value=True)
#     with col2:
#         show_chunks = st.checkbox("Show retrieved chunks")

#     if st.button("Ask"):
#         if not query.strip():
#             st.warning("Please enter a question")
#         else:
#             with st.spinner("Thinking..."):
#                 try:
#                     result = rag.query(
#                         question=query,
#                         top_k=top_k,
#                         use_cache=use_cache
#                     )

#                     st.success("Answer")
#                     st.write(result["answer"])

#                     st.divider()

#                     st.subheader("📄 Sources")
#                     for src in sorted(set(result["sources"])):
#                         st.markdown(f"- `{src}`")

#                     st.subheader("⚙️ Metadata")
#                     st.json({
#                         "chunks_retrieved": result["chunks_retrieved"],
#                         "cached": result.get("cached"),
#                         "response_time_ms": result["response_time_ms"],
#                         "tokens_used": result.get("tokens_used")
#                     })

#                     if show_chunks:
#                         st.divider()
#                         st.subheader("🔎 Retrieved Chunks")

#                         for i, chunk in enumerate(result.get("chunks", []), 1):
#                             with st.expander(
#                                 f"Chunk {i} | {chunk['document']['filename']} | similarity={chunk['similarity']:.3f}"
#                             ):
#                                 st.write(chunk["content"])

#                 except Exception as e:
#                     st.error(f"Query failed: {e}")

# # ==========================================================
# # 🧪 Debug / Inspect
# # ==========================================================
# with tab_debug:
#     st.subheader("🧪 System Debug")

#     if st.button("📊 Show Full System Stats"):
#         st.json(rag.get_stats())

#     st.divider()

#     st.caption("Use this tab for production debugging")

#     st.markdown("""
# **Helpful checks**
# - FAISS vectors > 0
# - Chunks > documents
# - Cache hits increasing on repeated queries
# """)


# ########################################################


import sys
import streamlit as st
from pathlib import Path
import tempfile
import json


# from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_system import rag
# from src.rag_system import rag

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="SmartRetriever Pro",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SmartRetriever Pro")
st.caption("Production-grade Retrieval Augmented Generation system")

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("⚙️ Retrieval Settings")

top_k = st.sidebar.slider(
    "Top-K Chunks",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Number of chunks retrieved from FAISS"
)

use_cache = st.sidebar.checkbox(
    "Use Cache",
    value=True
)

# ----------------------------
# Sidebar – System Status
# ----------------------------
with st.sidebar:
    st.header("⚙️ System Status")

    try:
        stats = rag.get_stats()

        st.success("System Online")

        st.metric("📄 Documents", stats["database"]["total_documents"])
        st.metric("🧩 Chunks", stats["database"]["total_chunks"])
        st.metric("🔢 FAISS Vectors", stats["faiss"]["total_vectors"])
        st.metric("⚡ Cache Keys", stats["cache"].get("total_keys", 0))

        if st.button("🗑️ Clear Cache"):
            rag.clear_cache()
            st.success("Cache cleared")

    except Exception as e:
        st.error("System not ready")
        st.caption(str(e))

# ----------------------------
# Tabs
# ----------------------------
# tab_upload, tab_query, tab_debug = st.tabs(
#     ["📤 Upload Documents", "🔍 Ask Query", "🧪 Debug / Inspect"]
# )

tab_upload, tab_query, tab_docs, tab_debug = st.tabs(
    ["📤 Upload Documents", "🔍 Ask Query", "📚 Documents", "🧪 Debug / Inspect"]
)


# ==========================================================
# 📤 Upload Documents
# ==========================================================
with tab_upload:
    st.subheader("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload text documents",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process Documents"):
            with st.spinner("Processing documents..."):
                results = []

                for file in uploaded_files:
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    try:
                        result = rag.add_document(
                            filepath=tmp_path,
                            filename=file.name
                        )
                        results.append(result)

                    except Exception as e:
                        results.append({
                            "success": False,
                            "filename": file.name,
                            "error": str(e)
                        })

                st.success("Document processing completed")

                st.subheader("📄 Results")
                st.json(results)

# ==========================================================
# 🔍 Query UI
# ==========================================================
with tab_query:
    st.subheader("🔍 Ask a Question")

    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What is Python?"
    )

    col1, col2 = st.columns(2)
    with col1:
        use_cache = st.checkbox("Use cache", value=True)
    with col2:
        show_chunks = st.checkbox("Show retrieved chunks")

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = rag.query(
                        question=query,
                        top_k=top_k,
                        use_cache=use_cache
                    )

                    st.success("Answer")
                    st.write(result["answer"])

                    st.divider()

                    st.subheader("📄 Sources")
                    for src in sorted(set(result["sources"])):
                        st.markdown(f"- `{src}`")

                    st.subheader("⚙️ Metadata")
                    st.json({
                        "chunks_retrieved": result["chunks_retrieved"],
                        "cached": result.get("cached"),
                        "response_time_ms": result["response_time_ms"],
                        "tokens_used": result.get("tokens_used")
                    })

                    if show_chunks:
                        st.divider()
                        st.subheader("🔎 Retrieved Chunks")

                        for i, chunk in enumerate(result.get("chunks", []), 1):
                            with st.expander(
                                f"Chunk {i} | {chunk['document']['filename']} | similarity={chunk['similarity']:.3f}"
                            ):
                                st.write(chunk["content"])

                except Exception as e:
                    st.error(f"Query failed: {e}")

# ==========================================================
# 📚 Documents Manager
# ==========================================================
# with tab_docs:
#     st.subheader("📚 Ingested Documents")

#     try:
#         documents = rag.db.get_all_documents()
#     except Exception as e:
#         st.error(f"Failed to load documents: {e}")
#         documents = []

#     if not documents:
#         st.info("No documents found in database.")
#     else:
#         # for doc in documents:
#         #     with st.container():
#         #         col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

#         #         col1.markdown(f"**📄 {doc.filename}**")
#         #         col2.markdown(f"ID: `{doc.id}`")
#         #         col3.markdown(f"Chunks: `{len(doc.chunks)}`")

#         #         if col4.button("🗑️", key=f"delete_{doc.id}"):
#         #             st.session_state["delete_doc_id"] = doc.id

#         for doc in documents:
#             doc_id = doc.get("id")
#             filename = doc.get("filename", "unknown")
#             chunks_count = doc.get("chunks", 0)

#             with st.container():
#                 col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

#                 col1.markdown(f"**📄 {filename}**")
#                 col2.markdown(f"ID: `{doc_id}`")
#                 col3.markdown(f"Chunks: `{chunks_count}`")

#                 if col4.button("🗑️", key=f"delete_{doc_id}"):
#                     st.session_state["delete_doc_id"] = doc_id

#             st.divider()


#     # -----------------------------
#     # Delete confirmation
#     # -----------------------------
#     if "delete_doc_id" in st.session_state:
#         doc_id = st.session_state["delete_doc_id"]

#         st.warning(
#             f"""
#             ⚠️ **Delete document ID {doc_id}?**

#             This will:
#             - Remove document + chunks from PostgreSQL
#             - NOT update FAISS automatically
#             - Require FAISS rebuild
#             """
#         )

#         col_yes, col_no = st.columns(2)

#         if col_yes.button("✅ Confirm Delete"):
#             rag.delete_document(doc_id)
#             del st.session_state["delete_doc_id"]
#             st.success("Document deleted. Please rebuild FAISS.")
#             st.experimental_rerun()

#         if col_no.button("❌ Cancel"):
#             del st.session_state["delete_doc_id"]
#             st.experimental_rerun()

#     st.divider()

#     # -----------------------------
#     # FAISS maintenance
#     # -----------------------------
#     st.subheader("🧱 Vector Index Maintenance")

#     if st.button("🔄 Rebuild FAISS Index"):
#         with st.spinner("Rebuilding FAISS index..."):
#             rag.rebuild_faiss_index()
#         st.success("FAISS index rebuilt successfully!")


# ==========================================================
# 📚 Documents Manager
# ==========================================================
with tab_docs:
    st.subheader("📚 Ingested Documents")

    try:
        documents = rag.db.get_all_documents()
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        documents = []

    if not documents:
        st.info("No documents found in database.")
    else:
        # for doc in documents:
        #     doc_id = doc.get("id")
        #     filename = doc.get("filename", "unknown")
        #     chunks_count = doc.get("chunks", 0)

        #     with st.container():
        #         col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

        #         col1.markdown(f"**📄 {filename}**")
        #         col2.markdown(f"ID: `{doc_id}`")
        #         col3.markdown(f"Chunks: `{chunks_count}`")

        #         if col4.button("🗑️", key=f"delete_{doc_id}"):
        #             st.session_state["delete_doc_id"] = doc_id


        # for doc in documents:
        #     doc_id = doc["id"]
        #     filename = doc["filename"]
        #     chunks_count = doc.get("chunks_count", 0)

        #     col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

        #     col1.markdown(f"**📄 {filename}**")
        #     col2.markdown(f"ID: `{doc_id}`")
        #     col3.markdown(f"Chunks: `{chunks_count}`")

        #     if col4.button("🗑️", key=f"delete_{doc_id}"):
        #         st.session_state["delete_doc_id"] = doc_id

        for doc in documents:
            doc_id = doc["id"]
            filename = doc["filename"]
            chunks_count = doc["chunks_count"]

            col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

            col1.markdown(f"**📄 {filename}**")
            col2.markdown(f"ID: `{doc_id}`")
            col3.markdown(f"Chunks: `{chunks_count}`")

            if col4.button("🗑️", key=f"delete_{doc_id}"):
                st.session_state["delete_doc_id"] = doc_id

            st.divider()

    # -----------------------------
    # Delete confirmation
    # -----------------------------
    if "delete_doc_id" in st.session_state:
        doc_id = st.session_state["delete_doc_id"]

        st.warning(
            f"""
                ⚠️ **Delete document ID {doc_id}?**

                This will:
                - Remove document + chunks from PostgreSQL
                - NOT update FAISS automatically
                - Require FAISS rebuild
                """
                        )

        col_yes, col_no = st.columns(2)

        if col_yes.button("✅ Confirm Delete"):
            rag.delete_document(doc_id)
            del st.session_state["delete_doc_id"]
            st.success("Document deleted. Please rebuild FAISS.")
            # st.experimental_rerun()
            st.rerun()

        if col_no.button("❌ Cancel"):
            del st.session_state["delete_doc_id"]
            # st.experimental_rerun()
            st.rerun()

    st.divider()

    # -----------------------------
    # FAISS maintenance
    # -----------------------------
    st.subheader("🧱 Vector Index Maintenance")

    if st.button("🔄 Rebuild FAISS Index"):
        with st.spinner("Rebuilding FAISS index..."):
            rag.rebuild_faiss_index()
        st.success("FAISS index rebuilt successfully!")


# ==========================================================
# 🧪 Debug / Inspect
# ==========================================================
with tab_debug:
    st.subheader("🧪 System Debug")

    if st.button("📊 Show Full System Stats"):
        st.json(rag.get_stats())

    st.divider()

    st.caption("Use this tab for production debugging")

    st.markdown("""
**Helpful checks**
- FAISS vectors > 0
- Chunks > documents
- Cache hits increasing on repeated queries
""")
