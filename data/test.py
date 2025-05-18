from langchain.agents import create_csv_agent
from custom_huggingface_llm import CustomHuggingFaceLLM

def call_csv_agent(question: str, file_path: str) -> str:
    try:
        # Instantiate the LLM
        model_path = "/exacto/abhishek/responsible_ai/models"
        phi4_llm = CustomHuggingFaceLLM(model_path=model_path)

        # Bind Python REPL tool
        tools = [{
            "name": "python_repl",
            "description": "Execute Python code to process CSV data using pandas."
        }]
        phi4_llm = phi4_llm.bind_tools(tools)

        # Create the CSV agent
        csv_agent = create_csv_agent(
            llm=phi4_llm,
            path=file_path,
            verbose=True,
            allow_dangerous_code=True,  # Required for Python REPL
            handle_parsing_errors=True  # Retry on parsing errors
        )

        # Run the query
        response = csv_agent.run(question)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    FILE_PATH = "us-500.csv"  # Adjust to your path if needed
    QUESTION = "Select the row where the first name is James and the last name contains Butt"
    response = call_csv_agent(QUESTION, FILE_PATH)
    print("Response:", response)
