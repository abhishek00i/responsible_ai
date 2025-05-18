# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
# from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
import os
# import uvicorn
# from langchain_community.callbacks import get_openai_callback
import pandas as pd
import sqlite3
from datetime import datetime
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
import logging
from logging.handlers import RotatingFileHandler
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="gemma3:12b") 
# load_dotenv()
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
# os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("ENDPOINT_URL")

session_id= "abc_123"
LOG_FILE = r"C:\Users\2393519\Documents\Azure\obj_agent.log"
MAX_LOG_SIZE = 100 * 1024 * 1024  # 20MB
logger = logging.getLogger("RotatingLogger")
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=100000)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("It is working")
 
store = {}  

def update_latest_user_message(session_id: str, new_message: str):
    """Replace the last AI-generated prompt template in chat history with the actual user input."""
    if session_id in store:
        messages = store[session_id].messages
 
        # Iterate in reverse to find the last human message and update it
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):  # Ensure it's a user message
               
                # Only replace if it contains the prompt template (heuristic: "Today's date is")
                if "Today's date is" in messages[i].content:
                    messages[i] = HumanMessage(content=new_message)
                    print(f"Updated latest user message for session {session_id}: {new_message}")
                    return True  # Stop after updating the first found user message
 
    logger.warning(f"No matching user message found to update in session {session_id}.")
    return False 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve chat history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
 
class DatabaseManager:
    """Class to manage database connections and operations."""
 
    def __init__(self, db_file: str, csv_folder: str):
        self.db_file = db_file
        self.csv_folder = csv_folder
 
    def initialize_database(self):
        """Initialize the database by loading CSV files into tables."""
        conn = sqlite3.connect(self.db_file)
        for file in os.listdir(self.csv_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(self.csv_folder, file)
                table_name = os.path.splitext(file)[0]
                df = pd.read_csv(file_path)
                df.to_sql(table_name, conn, index=False, if_exists='replace')
                print(f"Loaded {file} into {table_name} table.")
        conn.close()
 
 
class QueryHandler:
    """Class to handle SQL query execution."""
 
    def __init__(self, db_file: str):
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
 
    def run_query(self, query: str):
        """Execute the given SQL query and return the result."""
        try:
            result = self.db.run(query)
            logger.info(f"Executed query: {query}")
            return result
        
        except Exception as e:
            logger.error(f"Error executing query: {query}, Error: {e}")
            return str(e)
 
 
class AgentInitializer:
    """Class to initialize the LangChain agent."""
 
    def __init__(self, db: SQLDatabase):
        self.db = db
        self.llm = model
        self.table_info = db.get_table_info()
        self.prompt_template = self.create_prompt_template()
        self.tools = [Tool(
            name="query_database",
            func=self.run_query,
            description="Use this tool to execute SQL queries on the database."
        )]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            # agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            prompt= self.prompt_template
        )
        self.with_message_history = RunnableWithMessageHistory(self.llm, get_session_history)
 
    def create_prompt_template(self):
        """Create the prompt template for the LangChain agent."""
        return PromptTemplate.from_template("""
        Today's date is : {curr_date}
        You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the sqlite database.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        If the user sends a greeting, you also have to greet the user.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        To start you should ALWAYS look at the tables in the database to see what you can query.
        Then you should query the schema of the most relevant tables.
        If the user asks a question which is not related to your use case, do not answer it. Simply respond it with this is beyond my scope. Do not seek action after this thought, simply return this response.
        For example if a user asks for the prime minister's name, do not answer it.
        You may come across multiple parsing errors, make sure to handle parsing errors gracefully.  
        Do not loop the agent on missing action after thought, make sure to come up with your final answer as quickly as possible.
        Make sure to search the description and resolution notes column if you're not able to come up with an answer. These columns most likely will contain the keyword that the user has asked in the query. 
        Make sure to semantically analyze your response before generating the final answer. It should align with the relevance of the user query.                                                
        Learn from the previous responses of the user and provide the response. Take the chat history into account and learn the context of the query.
        ### Table names: {table_names}
        ### Table Info: {table_info}
        ### User question: {query}
        # Fetch the results from the database using the generated SQL query.
        """)
 
    def run_query(self, query: str):
        """Execute the given SQL query using the LangChain agent."""
        try:
            result = self.db.run(query)
            logger.info(f"Executed query: {query}")
            return result
        except Exception as e:
            logger.error(f"Error executing query: {query}, Error: {e}")
            return str(e)
 
 

db_file = "/Users/abhishek/Downloads/responsible_ai/outputs/data.db"
csv_folder = "/Users/abhishek/Downloads/responsible_ai/inputs"
# session_id = query_params.get("session_id", "default_session")
db_manager = DatabaseManager(db_file=db_file, csv_folder=csv_folder)
db_manager.initialize_database()

query_handler = QueryHandler(db_file=db_file)
agent_initializer = AgentInitializer(db=query_handler.db)

# Wrap the Agent with `RunnableWithMessageHistory`
agent_with_history = RunnableWithMessageHistory(
    agent_initializer.agent, get_session_history
)
 

  
data = "return phone numbers and email of James Butt."
chat_history = get_session_history(session_id)
 
            
response = agent_with_history.invoke(
    [HumanMessage(content=agent_initializer.prompt_template.format(
        curr_date=datetime.now().strftime("%Y-%m-%d"),
        table_names=query_handler.db.get_usable_table_names,
        query=data,
        table_info=agent_initializer.table_info
    ))],
    config={"configurable": {"session_id": session_id}},
)

final_response = response.get("output", "I couldn't generate a response.")
# Store input and final output in chat history
chat_history.add_user_message(data)
chat_history.add_ai_message(final_response)
update_latest_user_message(session_id, data)
