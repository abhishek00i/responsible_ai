import argparse
import os
import glob

try:
    from file_processing import load_documents, chunk_documents, create_and_store_faiss_index
    from retrieval import get_answer_from_documents
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure file_processing.py and retrieval.py are in the same directory or in PYTHONPATH.")
    # Define dummy functions so the script can be parsed and CLI help can be shown
    # The actual functionality will not work if imports fail.
    def load_documents(*args, **kwargs): print("Error: load_documents not available due to import error."); return []
    def chunk_documents(*args, **kwargs): print("Error: chunk_documents not available due to import error."); return []
    def create_and_store_faiss_index(*args, **kwargs): print("Error: create_and_store_faiss_index not available due to import error.")
    def get_answer_from_documents(*args, **kwargs): print("Error: get_answer_from_documents not available due to import error."); return "Unavailable due to import error."


SUPPORTED_EXTENSIONS = ['*.pdf', '*.txt', '*.docx', '*.xlsx', '*.html', '*.json', '*.png', '*.jpg', '*.jpeg']

def process_single_file(file_path: str):
    print(f"\nProcessing file: {file_path}...")
    
    # Check if core functions were actually imported
    if 'load_documents' not in globals() or \
       globals()['load_documents'].__doc__ is None and "unavailable" in globals()['load_documents'](check_only=True): # A bit of a hack to check if it's a dummy
        print(f"Core processing functions are not available, likely due to import errors. Skipping {file_path}.")
        return

    docs = load_documents(file_path)
    if not docs:
        # load_documents should print its own error or warning if a file is unsupported or not found
        print(f"No documents loaded from {file_path}. It might be empty, unsupported, or access was denied. Skipping further processing.")
        return

    chunks = chunk_documents(docs)
    if not chunks:
        print(f"Failed to chunk documents from {file_path}. Skipping.")
        return

    # create_and_store_faiss_index will print its own messages, including API key issues.
    create_and_store_faiss_index(file_path, chunks) 
    print(f"Completed processing attempt for: {file_path}")

def handle_process_command(args):
    path = args.path
    print(f"Process command called with path: '{path}'")

    if os.path.isdir(path):
        print(f"Processing directory: {path}")
        found_files = False
        for ext in SUPPORTED_EXTENSIONS:
            search_pattern = os.path.join(path, ext)
            for filepath in glob.glob(search_pattern):
                found_files = True
                process_single_file(filepath)
        if not found_files:
            print(f"No supported files found in directory: {path}")
    elif os.path.isfile(path):
        # Check if the file extension is one of the supported ones for individual file processing
        file_extension = '*' + os.path.splitext(path)[1].lower()
        is_supported = any(file_extension == ext.lower() for ext in SUPPORTED_EXTENSIONS)
        if is_supported:
            process_single_file(path)
        else:
            print(f"File type {os.path.splitext(path)[1]} is not directly supported for processing. Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}")
            print(f"If you believe this file type should be processed by one of the loaders (e.g. a .log as .txt), please ensure it's handled by the underlying 'load_documents' function.")
    else:
        print(f"Invalid path: '{path}'. Please provide a valid file or directory.")

def handle_ask_command(args):
    file_path = args.file
    query = args.query
    print(f"\nAsk command called for file: '{file_path}' with query: '{query}'")
    
    if 'get_answer_from_documents' not in globals() or \
       globals()['get_answer_from_documents'].__doc__ is None and "unavailable" in globals()['get_answer_from_documents'](check_only=True):
        print("Error: get_answer_from_documents function is not available, likely due to import errors.")
        return

    if not os.path.isfile(file_path):
        print(f"Error: File not found at '{file_path}'. Please ensure the path is correct.")
        return

    # get_answer_from_documents will print its own messages regarding API key issues or if no answer is found.
    answer = get_answer_from_documents(query, file_path)
    
    print("\n---")
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print("---")

# Dummy function modification for check_only to avoid print during dummy function call
def _make_dummy_unavailable(name):
    def dummy_func(*args, **kwargs):
        if kwargs.get('check_only'): return "unavailable"
        print(f"Error: {name} not available due to import error.")
        if name == "load_documents" or name == "chunk_documents": return []
        if name == "get_answer_from_documents": return "Unavailable due to import error."
    return dummy_func

def main():
    # Redefine dummy functions if imports failed, to allow CLI help to work better
    # And to make the check in process_single_file more robust
    if 'load_documents' not in globals() or globals()['load_documents'].__module__ == __name__: # Check if it's a dummy
        global load_documents, chunk_documents, create_and_store_faiss_index, get_answer_from_documents
        load_documents = _make_dummy_unavailable('load_documents')
        chunk_documents = _make_dummy_unavailable('chunk_documents')
        create_and_store_faiss_index = _make_dummy_unavailable('create_and_store_faiss_index')
        get_answer_from_documents = _make_dummy_unavailable('get_answer_from_documents')


    parser = argparse.ArgumentParser(
        description="CLI tool to process documents and ask questions using a RAG model.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve formatting in help messages
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands. Use <command> --help for more details.")

    # Process subparser
    process_parser = subparsers.add_parser("process", 
                                           help="Process a file or a directory of files to create FAISS indexes.",
                                           description="This command processes the specified file(s).\n"
                                                       "For each supported file, it loads the content, chunks it, and then attempts\n"
                                                       "to create and store a FAISS vector index in the 'faiss_indexes' directory.\n"
                                                       "Note: Creating embeddings for the index requires a valid OPENAI_API_KEY.")
    process_parser.add_argument("path", type=str, help="The path to a single file or a directory containing files to process.")
    process_parser.set_defaults(func=handle_process_command)

    # Ask subparser
    ask_parser = subparsers.add_parser("ask", 
                                       help="Ask a question about a specific processed file using its content and FAISS index.",
                                       description="This command attempts to answer a query based on the content of the specified file.\n"
                                                   "It uses a combination of semantic search (if a FAISS index exists for the file)\n"
                                                   "and keyword search on the document's content. The retrieved information is then\n"
                                                   "passed to an LLM to generate an answer.\n"
                                                   "Note: This command requires a valid OPENAI_API_KEY for both FAISS index loading (if used)\n"
                                                   "and for the LLM to generate an answer.")
    ask_parser.add_argument("--file", "-f", type=str, required=True, help="The path to the specific file to ask questions against.")
    ask_parser.add_argument("--query", "-q", type=str, required=True, help="The question to ask.")
    ask_parser.set_defaults(func=handle_ask_command)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        # If no command is given, or if imports failed preventing command execution
        if 'load_documents' in globals() and hasattr(globals()['load_documents'], '__doc__') and \
           globals()['load_documents'].__doc__ is None and "unavailable" in globals()['load_documents'](check_only=True):
            print("\nOne or more core modules could not be imported. Please check the initial error messages.")
            print("Displaying help as fallback:")
        parser.print_help()

if __name__ == '__main__':
    main()
