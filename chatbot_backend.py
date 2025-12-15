from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# -------------------
# MCP Integration
# -------------------
# Define server parameters for our local mcp_server.py
server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"],
    env=None
)

async def get_mcp_tools():
    """Connect to MCP server and load tools."""
    # Note: In a production app you'd manage the lifecycle (connect/disconnect)
    # more carefully. For this demo, we'll assume a fresh connection or
    # that the adapter handles it. 
    # Actually, load_mcp_tools is an async context manager or requires one.
    # To keep it simple in this synchronous-ish graph setup, we might need a workaround
    # or ensure we run this in an async context.
    # 
    # However, LangGraph nodes are often async. Let's try to load them at startup
    # if possible, OR load them dynamically.
    #
    # Because `load_mcp_tools` is async, and we are at module level, 
    # we can't easily await it here without an event loop.
    # 
    # workaround: fastmcp server tools are static in this case.
    # But strictly speaking we should connect.
    #
    # Let's try a lazy loading approach or just define them if we can. 
    # simpler: Just use the `mcp_tools` list if we can initialize it.
    
    # We will wrap the connection in a way that the tools are available.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await load_mcp_tools(session)

# FIXME: This is tricky because `tools` list is passed to `bind_tools` at startup.
# We need the tools available synchronously or we need a way to refresh them.
# The simplest approach related to the "langchain-mcp-adapters" usage 
# usually involves an async context.
#
# Alternative: Since we controls the server, we *know* the tools. 
# But the point is to use the protocol.
# 
# We'll create a wrapper tool that connects on demand? 
# Or we can try to run the async loader once using `asyncio.run`.
import asyncio
try:
    mcp_tools = asyncio.run(get_mcp_tools())
except Exception as e:
    print(f"Failed to load MCP tools: {e}")
    mcp_tools = []


tools = [search_tool, get_stock_price, calculator, rag_tool] + mcp_tools
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


def generate_summary_title(thread_id: str, messages: list[BaseMessage]) -> str:
    """
    Generate a short 4-5 word title based on the first few user messages.
    """
    human_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        return "New Chat"

    # Take first 4 messages
    context = "\n".join(human_messages[:4])
    prompt = (
        f"Summarize the following conversation start into a short title (4-5 words max). "
        f"Do not use quotes. Just the title.\n\nConversation:\n{context}"
    )
    
    try:
        response = llm.invoke(prompt)
        title = response.content.strip().replace('"', '')
        return title
    except Exception:
        return "New Chat"


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    """
    Retrieve all thread IDs and their titles (generated or stored).
    """
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    cursor = conn.cursor()
    
    # Ensure summaries table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_summaries (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)
    conn.commit()

    # Get all threads from checkpoints
    all_threads_ids = set()
    for checkpoint in checkpointer.list(None):
        all_threads_ids.add(checkpoint.config["configurable"]["thread_id"])
    
    results = []
    
    for tid in all_threads_ids:
        # Check if we have a title
        cursor.execute("SELECT title FROM thread_summaries WHERE thread_id = ?", (tid,))
        row = cursor.fetchone()
        
        if row:
            title = row[0]
        else:
            # Generate title if missing
            # Fetch state to get messages
            state = chatbot.get_state({"configurable": {"thread_id": tid}})
            messages = state.values.get("messages", [])
            
            # If no messages, it's an empty/new chat
            if not messages:
                title = "New Chat"
            else:
                title = generate_summary_title(tid, messages)
            
            # Save to DB
            cursor.execute(
                "INSERT OR REPLACE INTO thread_summaries (thread_id, title) VALUES (?, ?)",
                (tid, title)
            )
            conn.commit()
            
        results.append({"id": tid, "title": title})
        
    conn.close()
    return results


def delete_thread_data(thread_id: str):
    """
    Delete a thread and its summary from the database.
    """
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    cursor = conn.cursor()
    
    # Delete summary
    cursor.execute("DELETE FROM thread_summaries WHERE thread_id = ?", (thread_id,))
    
    # Delete from checkpoints (LangGraph specific tables)
    # Note: Depending on LangGraph version, table names might vary (checkpoints, writes).
    # We will attempt to delete from the standard tables if they exist.
    try:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    except Exception as e:
        print(f"Error clean up checkpoints: {e}")

    conn.commit()
    conn.close()
    return True


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})         