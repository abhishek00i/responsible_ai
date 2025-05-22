import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredImageLoader,
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import shutil # For cleaning up the faiss_indexes directory

def load_documents(file_path: str) -> list[Document]:
    file_extension = os.path.splitext(file_path)[1].lower()
    documents = []

    loader_map = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.html': UnstructuredHTMLLoader,
        '.json': JSONLoader,
        '.png': UnstructuredImageLoader,
        '.jpg': UnstructuredImageLoader,
        '.jpeg': UnstructuredImageLoader,
    }

    try:
        if file_extension in loader_map:
            loader_class = loader_map[file_extension]
            if file_extension == '.json':
                loader = loader_class(file_path, jq_schema='.', text_content=False)
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
        else:
            print(f"Warning: Unsupported file type: {file_extension} for file {file_path}")
            return []
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []
        
    return documents

def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

def create_and_store_faiss_index(file_path: str, documents: list[Document], embeddings_model_name: str = "text-embedding-ada-002") -> None:
    if not documents:
        print(f"No documents provided to create FAISS index for {file_path}")
        return

    try:
        print(f"Initializing embeddings model: {embeddings_model_name}...")
        embeddings = OpenAIEmbeddings(model=embeddings_model_name)
        
        print(f"Creating FAISS vector store for {file_path}...")
        vector_store = FAISS.from_documents(documents, embeddings)
        
        faiss_dir = "faiss_indexes"
        os.makedirs(faiss_dir, exist_ok=True)
        
        base_name = os.path.basename(file_path)
        index_name = os.path.splitext(base_name)[0]
        
        print(f"Saving FAISS index to {os.path.join(faiss_dir, index_name)}...")
        vector_store.save_local(folder_path=faiss_dir, index_name=index_name)
        print(f"FAISS index for {file_path} created and saved successfully as {index_name} in {faiss_dir}/.")
        
    except Exception as e:
        print(f"Error creating or saving FAISS index for {file_path}: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly and you have an internet connection.")

if __name__ == '__main__':
    dummy_files_created = []
    faiss_index_dir_main = "faiss_indexes" # Define at a broader scope for cleanup

    try:
        # Create dummy files
        with open("test.txt", "w") as f:
            f.write("This is a test text file. It is a bit longer to see if chunking might happen, but probably not with default chunk size. Let's add more content. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
        dummy_files_created.append("test.txt")

        with open("test.json", "w") as f:
            f.write('{"name": "test", "data": "some json content", "nested": {"value": 123}}')
        dummy_files_created.append("test.json")

        # Test load_documents
        print("Testing with test.txt:")
        txt_docs = load_documents("test.txt")
        if txt_docs:
            print(f"Loaded {len(txt_docs)} documents from test.txt. Content: '{txt_docs[0].page_content[:100]}...'")
        else:
            print("No documents loaded from test.txt")
        
        print("\nTesting with test.json:")
        json_docs = load_documents("test.json")
        if json_docs:
            print(f"Loaded {len(json_docs)} documents from test.json. Type of content: {type(json_docs[0].page_content)}. Content: '{json_docs[0].page_content}'")
        else:
            print("No documents loaded from test.json")

        print("\nTesting with non_existent_file.pdf:")
        pdf_docs = load_documents("non_existent_file.pdf")
        print(f"Loaded {len(pdf_docs)} documents from non_existent_file.pdf")

        print("\nTesting with unsupported_file.xyz:")
        xyz_docs = load_documents("unsupported_file.xyz")
        print(f"Loaded {len(xyz_docs)} documents from unsupported_file.xyz")

        # Test chunk_documents
        chunked_txt_docs = [] # Initialize in case txt_docs is empty
        if txt_docs: 
            print(f"\nOriginal number of documents from text.txt: {len(txt_docs)}")
            chunked_txt_docs = chunk_documents(txt_docs, chunk_size=100, chunk_overlap=20) 
            print(f"Number of chunked documents from test.txt: {len(chunked_txt_docs)}")
            if chunked_txt_docs:
                print(f"Content of the first chunk from test.txt: '{chunked_txt_docs[0].page_content}'")
                if len(chunked_txt_docs) > 1:
                    print(f"Content of the second chunk from test.txt: '{chunked_txt_docs[1].page_content}'")

        print("\nTesting chunking with a small document (smaller than default chunk size):")
        small_document_content = "This is a short sentence."
        small_document = [Document(page_content=small_document_content)]
        chunked_small_doc = chunk_documents(small_document)
        print(f"Number of chunks for small document: {len(chunked_small_doc)}")
        if chunked_small_doc:
            print(f"Content of first chunk for small document: '{chunked_small_doc[0].page_content}'")
            assert chunked_small_doc[0].page_content == small_document_content 

        print("\nTesting chunking with an empty list of documents:")
        empty_chunks = chunk_documents([])
        print(f"Number of chunks for empty list: {len(empty_chunks)}")
        assert len(empty_chunks) == 0
        
        # Test create_and_store_faiss_index
        if chunked_txt_docs:
            print("\nTesting FAISS index creation for test.txt...")
            create_and_store_faiss_index("test.txt", chunked_txt_docs)
            
            faiss_index_path = os.path.join(faiss_index_dir_main, "test.faiss")
            faiss_pkl_path = os.path.join(faiss_index_dir_main, "test.pkl")
            
            if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
                print(f"FAISS index files found at {faiss_index_path} and {faiss_pkl_path}")
            else:
                # This branch is expected if OPENAI_API_KEY is missing/invalid
                print(f"FAISS index files NOT found. This is expected if OPENAI_API_KEY is missing or invalid.")

        print("\nTesting FAISS index creation with empty documents:")
        create_and_store_faiss_index("empty_test.txt", [])

    finally:
        # Clean up dummy files
        for dummy_file in dummy_files_created:
            if os.path.exists(dummy_file):
                os.remove(dummy_file)
        print("\nCleaned up dummy files.")

        # Clean up faiss_indexes directory
        if os.path.exists(faiss_index_dir_main):
            print(f"Cleaning up {faiss_index_dir_main} directory...")
            shutil.rmtree(faiss_index_dir_main)
        print(f"Cleaned up {faiss_index_dir_main} directory (if it existed).")
