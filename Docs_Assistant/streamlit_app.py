import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from src.search import RagSearch

# Initialize RAG Search (loads vector store or builds if needed)
@st.cache_resource
def load_rag():
    return RagSearch("./data/vector_store")

rag_search = load_rag()

# Streamlit UI
st.title("📚 Document Assistant - RAG Search")
st.markdown("Ask questions about your documents and get AI-powered answers!")

# Query input
query = st.text_input("Enter your question:", placeholder="e.g., What is Attention Mechanism?")

if st.button("Search", type="primary"):
    if query.strip():
        with st.spinner("Searching and summarizing..."):
            try:
                summary = rag_search.search_summarize(query)
                st.success("Answer found!")
                st.markdown("### Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain, FAISS, and Groq*")