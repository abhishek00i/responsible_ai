from langchain_experimental.agents import create_csv_agent
from custom_huggingface_llm import CustomHuggingFaceLLM
import asyncio
import json

async def call_csv_agent(question: str, file_path: str, stream: bool = False) -> str:
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
        )

        if stream:
            # Streaming response
            async for chunk in csv_agent.astream({"input": question}):
                if isinstance(chunk, dict) and "output" in chunk:
                    print(chunk["output"], end="", flush=True)
                elif isinstance(chunk, dict):
                    print(json.dumps(chunk), end="", flush=True)
                else:
                    print(chunk, end="", flush=True)
            return ""
        else:
            # Non-streaming response using ainvoke
            response = await csv_agent.ainvoke({"input": question})
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            elif isinstance(response, dict):
                return json.dumps(response)
            else:
                return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    FILE_PATH = "us-500.csv"  # Adjust to your path if needed
    QUESTION = "Select the row where the first_name is James and the last_name contains Butt"

    # Run with streaming
    print("Streaming Response:")
    asyncio.run(call_csv_agent(QUESTION, FILE_PATH, stream=True))

    # Run without streaming
    print("\nNon-Streaming Response:")
    response = asyncio.run(call_csv_agent(QUESTION, FILE_PATH, stream=False))
    print("Response:", response)
