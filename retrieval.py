import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import shutil # For test cleanup
import re # For keyword search
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Assuming file_processing.py is in the same directory or PYTHONPATH is configured
try:
    from file_processing import load_documents, chunk_documents, create_and_store_faiss_index
except ImportError:
    print("Warning: Could not import from file_processing.py. Test block might fail.")
    # Define dummy functions if needed for the script to be parsable, though test will fail
    def load_documents(*args, **kwargs): return [] # Return empty list
    def chunk_documents(*args, **kwargs): return [] # Return empty list
    def create_and_store_faiss_index(*args, **kwargs): pass

def load_faiss_index(folder_path: str, index_name: str, embeddings_model_name: str = "text-embedding-ada-002") -> FAISS | None:
    try:
        actual_faiss_path = os.path.join(folder_path, index_name + ".faiss")
        actual_pkl_path = os.path.join(folder_path, index_name + ".pkl")

        if not (os.path.exists(actual_faiss_path) and os.path.exists(actual_pkl_path)):
            print(f"Error: Index files not found at {actual_faiss_path} or {actual_pkl_path}")
            return None

        # print(f"Initializing embeddings model: {embeddings_model_name} for loading...")
        embeddings = OpenAIEmbeddings(model=embeddings_model_name)
        
        # print(f"Loading FAISS index '{index_name}' from {folder_path}...")
        vector_store = FAISS.load_local(
            folder_path=folder_path, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True 
        )
        # print(f"FAISS index '{index_name}' loaded successfully from {folder_path}.")
        return vector_store
        
    except Exception as e:
        # print(f"Error loading FAISS index '{index_name}' from {folder_path}: {e}")
        # print("Please ensure your OPENAI_API_KEY is set correctly, you have an internet connection, and the index files are valid.")
        return None

def hybrid_search(query: str, faiss_index: FAISS | None, documents: list[Document], top_k_semantic: int = 3, top_k_keyword: int = 3) -> list[Document]:
    if not query:
        print("Warning: Empty query for hybrid search.")
        return []
    
    if not faiss_index and not documents:
        print("Warning: Both FAISS index and documents list are missing for hybrid search.")
        return []

    semantic_docs = []
    if faiss_index:
        try:
            # print(f"Performing semantic search for: '{query}' (top_k={top_k_semantic})")
            results_with_scores = faiss_index.similarity_search_with_score(query, k=top_k_semantic)
            semantic_docs = [doc for doc, score in results_with_scores]
            # print(f"Found {len(semantic_docs)} semantic results.")
        except Exception as e:
            print(f"Error during semantic search: {e}. This might be due to missing API key or invalid index.")
    else:
        # print("No FAISS index provided or available for semantic search.")
        pass


    keyword_docs = []
    if documents:
        # print(f"Performing keyword search for: '{query}' in {len(documents)} documents.")
        query_keywords = [keyword.lower() for keyword in query.split() if keyword]
        
        if not query_keywords:
            # print("No valid keywords extracted from query for keyword search.")
            pass
        else:
            for doc in documents:
                doc_content_lower = doc.page_content.lower()
                if any(re.search(r"\b" + re.escape(keyword) + r"\b", doc_content_lower) for keyword in query_keywords):
                    keyword_docs.append(doc)
            # print(f"Found {len(keyword_docs)} keyword results (pre-deduplication).")
    else:
        # print("No documents provided for keyword search.")
        pass

    combined_results = []
    seen_contents = set()
    for doc in semantic_docs:
        if doc.page_content not in seen_contents:
            combined_results.append(doc)
            seen_contents.add(doc.page_content)
    for doc in keyword_docs:
        if doc.page_content not in seen_contents:
            combined_results.append(doc)
            seen_contents.add(doc.page_content)
            
    # print(f"Total combined and de-duplicated results: {len(combined_results)}")
    return combined_results

def get_answer_from_documents(query: str, file_path: str, faiss_index_dir: str = "faiss_indexes", 
                              embeddings_model_name: str = "text-embedding-ada-002", 
                              llm_model_name: str = "gpt-3.5-turbo") -> str:
    try:
        print(f"\nAttempting to answer query: '{query}' using file: '{file_path}'")

        index_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print("Loading FAISS index...")
        loaded_faiss_index = load_faiss_index(folder_path=faiss_index_dir, index_name=index_name, embeddings_model_name=embeddings_model_name)
        if loaded_faiss_index is None:
            # load_faiss_index prints its own errors, but we add context here
            print(f"FAISS index '{index_name}' could not be loaded from '{faiss_index_dir}'. Semantic search will be skipped.")
        # If index loading fails, hybrid_search will proceed with keyword search only if documents are available.

        print("Loading and chunking document for context and keyword search...")
        original_docs = load_documents(file_path) 
        if not original_docs:
            # load_documents prints FileNotFoundError, this is for other cases or if it returns empty
            return f"Could not load the document content from {file_path}. Please ensure the file exists and is accessible."
        
        chunked_original_docs = chunk_documents(original_docs)
        if not chunked_original_docs:
            return f"Could not process document {file_path} into chunks."

        print("Performing hybrid search...")
        retrieved_chunks = hybrid_search(query, loaded_faiss_index, chunked_original_docs)
        
        if not retrieved_chunks:
            return "No relevant information found in the document for your query."

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_chunks])
        
        print("Initializing LLM and creating RAG chain...")
        llm = ChatOpenAI(model_name=llm_model_name)
        
        prompt_template_str = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer based on the context, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Answer:"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        
        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()} 
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("Invoking RAG chain to get answer...")
        answer = rag_chain.invoke(query)
        
        return answer

    except Exception as e:
        print(f"An error occurred in get_answer_from_documents: {e}")
        error_message = "An error occurred while trying to generate an answer. "
        if "OPENAI_API_KEY" in str(e) or "api_key" in str(e):
            error_message += "This might be due to a missing or invalid OpenAI API key. "
        error_message += "Please ensure your API key is correctly configured, has an active subscription, and models are available."
        return error_message

if __name__ == '__main__':
    TEST_FILE_NAME = "test_retrieval_doc.txt"
    FAISS_DIR_FOR_CREATION = "faiss_indexes" 
    INDEX_NAME = os.path.splitext(TEST_FILE_NAME)[0]
    loaded_index: FAISS | None = None 

    with open(TEST_FILE_NAME, "w") as f:
        f.write("This is a test document for FAISS loading and retrieval. It contains some text about Langchain and FAISS to be indexed for semantic capabilities. Langchain helps build LLM applications. FAISS is for efficient similarity search.")
    print(f"Created dummy file: {TEST_FILE_NAME}")

    print(f"\n--- Preparing for FAISS index (load_faiss_index test) ---")
    print("NOTE: The following steps (create_and_store_faiss_index, and parts of load_faiss_index) require a valid OPENAI_API_KEY.")
    docs_for_faiss = load_documents(TEST_FILE_NAME)
    chunked_docs_for_faiss = []
    if docs_for_faiss:
        chunked_docs_for_faiss = chunk_documents(docs_for_faiss)
        if chunked_docs_for_faiss:
            create_and_store_faiss_index(TEST_FILE_NAME, chunked_docs_for_faiss) # From file_processing
            print(f"\nAttempting to load FAISS index '{INDEX_NAME}' from '{FAISS_DIR_FOR_CREATION}'...")
            loaded_index = load_faiss_index(folder_path=FAISS_DIR_FOR_CREATION, index_name=INDEX_NAME)
            if loaded_index:
                print("load_faiss_index test: Successfully loaded the FAISS index.")
            else:
                print("load_faiss_index test: Failed to load the FAISS index (expected if OPENAI_API_KEY is missing or index creation failed).")
        else:
            print("Skipping FAISS creation/load test as document chunking failed.")
    else:
        print("Skipping FAISS creation/load test as document loading failed.")

    print("\nTesting load_faiss_index with a non-existent index name...")
    non_existent_result = load_faiss_index(folder_path=FAISS_DIR_FOR_CREATION, index_name="this_index_does_not_exist")
    if not non_existent_result:
        print(f"load_faiss_index test: Correctly handled non-existent index name. Output was None.")
    
    print("\nTesting load_faiss_index with a non-existent folder...")
    non_existent_folder_result = load_faiss_index(folder_path="this_folder_does_not_exist", index_name=INDEX_NAME)
    if not non_existent_folder_result:
        print(f"load_faiss_index test: Correctly handled non-existent folder. Output was None.")

    # --- Testing Hybrid Search ---
    print("\n--- Testing Hybrid Search ---")
    doc1_content = "The quick brown fox jumps over the lazy dog."
    doc2_content = "Langchain provides tools for building language model applications."
    doc3_content = "FAISS is a library for efficient similarity search. This fox is fast."
    doc4_content = "OpenAI models like GPT-3 can be used with Langchain."
    test_hybrid_documents = [
        Document(page_content=doc1_content, metadata={"source": "hybrid_test_doc", "id": 1}),
        Document(page_content=doc2_content, metadata={"source": "hybrid_test_doc", "id": 2}),
        Document(page_content=doc3_content, metadata={"source": "hybrid_test_doc", "id": 3}),
        Document(page_content=doc4_content, metadata={"source": "hybrid_test_doc", "id": 4})
    ]
    hybrid_query_1 = "efficient language model applications with FAISS"
    print(f"\nTest 1: Hybrid search with query: '{hybrid_query_1}'")
    hybrid_results_1 = hybrid_search(hybrid_query_1, loaded_index, test_hybrid_documents)
    # ... (rest of hybrid search tests from previous step, slightly condensed for brevity here) ...
    print(f"Hybrid search 1 yielded {len(hybrid_results_1)} results.")

    hybrid_query_2 = "brown fox"
    print(f"\nTest 2: Hybrid search with query: '{hybrid_query_2}'")
    hybrid_results_2 = hybrid_search(hybrid_query_2, loaded_index, test_hybrid_documents)
    print(f"Hybrid search 2 yielded {len(hybrid_results_2)} results.")
    expected_keyword_docs_for_query2 = {doc1_content, doc3_content}
    found_keyword_docs_for_query2 = sum(1 for doc in hybrid_results_2 if doc.page_content in expected_keyword_docs_for_query2)
    if found_keyword_docs_for_query2 == len(expected_keyword_docs_for_query2):
         print("Keyword search part for 'brown fox' correctly found all expected documents.")
    else:
         print(f"Keyword search part for 'brown fox' found {found_keyword_docs_for_query2} of {len(expected_keyword_docs_for_query2)} expected documents.")
    
    print(f"\nTest 3: Hybrid search with empty query")
    hybrid_results_3 = hybrid_search("", loaded_index, test_hybrid_documents)
    assert len(hybrid_results_3) == 0

    print(f"\nTest 5: Hybrid search with query '{hybrid_query_2}', no FAISS index, only keyword documents")
    hybrid_results_5 = hybrid_search(hybrid_query_2, None, test_hybrid_documents) # Explicitly None for FAISS
    found_keyword_docs_for_query5 = sum(1 for doc in hybrid_results_5 if doc.page_content in expected_keyword_docs_for_query2)
    if found_keyword_docs_for_query5 == len(expected_keyword_docs_for_query2):
        print("Test 5: Correctly found documents with 'brown fox' via keyword search when FAISS index is None.")
    else:
        print(f"Test 5: Did not find all expected documents with 'brown fox'. Found {found_keyword_docs_for_query5}.")


    # --- Testing get_answer_from_documents ---
    print("\n--- Testing get_answer_from_documents ---")
    print("NOTE: This test requires a valid OPENAI_API_KEY to be set in the environment for all parts to function (FAISS loading, LLM).")
    print(f"It will use the file '{TEST_FILE_NAME}' which should have been created.")
    
    # Ensure TEST_FILE_NAME exists if previous steps failed to create it
    if not os.path.exists(TEST_FILE_NAME):
        with open(TEST_FILE_NAME, "w") as f:
            f.write("This is a test document about Langchain and FAISS. Langchain helps build LLM apps. FAISS is for search.")
        print(f"Re-created {TEST_FILE_NAME} for get_answer_from_documents test.")

    test_query_for_answer = "What is Langchain?"
    answer = get_answer_from_documents(test_query_for_answer, TEST_FILE_NAME, faiss_index_dir=FAISS_DIR_FOR_CREATION)
    print(f"\nQuery: {test_query_for_answer}")
    print(f"Answer: {answer}")

    test_query_no_info = "What is the color of the sky in this document?"
    answer_no_info = get_answer_from_documents(test_query_no_info, TEST_FILE_NAME, faiss_index_dir=FAISS_DIR_FOR_CREATION)
    print(f"\nQuery: {test_query_no_info}")
    print(f"Answer: {answer_no_info}")
    
    answer_no_file = get_answer_from_documents("Any query", "non_existent_file_for_get_answer.txt")
    print(f"\nQuery for non-existent file: Any query")
    print(f"Answer: {answer_no_file}")

    # Clean up
    print("\nCleaning up test files and directories...")
    if os.path.exists(TEST_FILE_NAME):
        os.remove(TEST_FILE_NAME)
        print(f"Removed dummy file: {TEST_FILE_NAME}")
    
    if os.path.exists(FAISS_DIR_FOR_CREATION):
        shutil.rmtree(FAISS_DIR_FOR_CREATION)
        print(f"Removed directory: {FAISS_DIR_FOR_CREATION}")
    else:
        print(f"Directory {FAISS_DIR_FOR_CREATION} not found, no need to remove (this is expected if index creation failed).")
    
    print("Cleanup complete.")
