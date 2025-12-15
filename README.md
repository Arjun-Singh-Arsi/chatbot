# 🤖 Smart-Agent Chatbot

An advanced, full-stack AI agent built with **Streamlit** and **LangGraph**. This chatbot goes beyond simple text generation—it is an autonomous agent capable of using tools, remembering conversation history, and performing Retrieval-Augmented Generation (RAG) on uploaded documents.

---

## 🚀 Key Features

- **🧠 Agentic Workflow**: Powered by `LangGraph`'s StateGraph, allowing the AI to loop, self-correct, and decide when to use specific tools.
- **📚 RAG (Retrieval-Augmented Generation)**: Upload PDF documents to instantly index them using `FAISS`. The agent can then answer specific questions based on the document's content.
- **💾 Persistent Memory**: Uses `SQLite` to store conversation history. Users can switch between different chat threads and resume distinct conversations at any time.
- **🛠️ Integrated Tools**:
  - **Web Search**: Real-time information using DuckDuckGo.
  - **Stock Prices**: Live market data via Alpha Vantage.
  - **Calculator**: Robust mathematical operations.
  - **MCP Integration**: Extensible architecture designed to support the Model Context Protocol.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (UI, Session State, Streaming)
- **Backend**: [LangGraph](https://python.langchain.com/docs/langgraph) & [LangChain](https://python.langchain.com/) (Orchestration, Tooling)
- **LLM**: OpenAI GPT-4o-mini
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Database**: SQLite (for chat history and checkpointers)

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Arjun-Singh-Arsi/chatbot.git
    cd chatbot
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv cbenv
    # Windows
    .\cbenv\Scripts\activate
    # Mac/Linux
    source cbenv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add your API keys:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    # Optional: other generic keys if needed
    ```

5.  **Run the Application**
    ```bash
    streamlit run chatbot_frontend.py
    ```

---

## 📂 Project Structure

- `chatbot_frontend.py`: The Streamlit-based user interface. Handles user input, file uploads, and rendering of the chat stream.
- `chatbot_backend.py`: The core logic containing the LangGraph agent, tool definitions, and RAG implementation.
- `mcp_server.py`: Server script for Model Context Protocol integration.
- `chatbot.db`: Local SQLite database for storing conversation persistence (created at runtime).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
