
### Key Changes in `custom_huggingface_llm.py`
1. **Improved `_parse_react_output`**:
   - Enhanced the query parsing logic to correctly handle conditions like `first_name is James` and `last_name contains Butt`.
   - Generates valid pandas code by constructing a `result` variable before printing.
   - Uses `str.contains` with `na=False` to handle potential NaN values in the DataFrame.
   - Properly escapes newlines in the `action_input` to ensure valid JSON.
2. **Maintained Streaming Support**:
   - The `_astream` method is retained as it was, as it already supports streaming and can be used in the updated main file.
3. **Robust Condition Parsing**:
   - Handles multiple conditions (e.g., `first_name is James and last_name contains Butt`) by splitting on `and` and building a combined pandas condition.
   - Supports both `is` and `=` for equality checks to make parsing more flexible.

### Updated Code for `test.py` (Main File)

```python
from langchain_experimental.agents import create_csv_agent
from custom_huggingface_llm import CustomHuggingFaceLLM
import asyncio

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
                if chunk.get("output"):
                    print(chunk["output"], end="", flush=True)
            return ""
        else:
            # Non-streaming response using invoke
            response = await csv_agent.ainvoke({"input": question})
            return response.get("output", f"Error: No output in response")
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
