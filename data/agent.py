from langchain.agents import create_csv_agent
from langchain_core.tools import Tool

# Instantiate the LLM
model_path = "/exacto/abhishek/responsible_ai/models"
phi4_llm = CustomHuggingFaceLLM(model_path=model_path)

# Define tools (if needed)
tools = [
    {
        "name": "csv_reader",
        "description": "Reads and processes CSV data",
        "schema": {"type": "object", "properties": {"query": {"type": "string"}}}
    }
]

# Bind tools to the LLM
phi4_llm = phi4_llm.bind_tools(tools)

# Create the CSV agent
agent = create_csv_agent(
    llm=phi4_llm,
    path="data.csv",
    verbose=True,
    allow_dangerous_code=True  # Required for CSV agent to execute code
)

# Run a query
response = agent.run("How many rows are in the CSV file?")
print(response)
