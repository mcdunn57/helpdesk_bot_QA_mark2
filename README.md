# Helpdesk QA Bot mk3.5


9Dec25: updated python code to use a streamlit front end. Version to use right now is ...3.5.py

Use the following to run py when in the project directory: 
streamlit run /Users/mike/Documents/python_stuff/helpdesk_bot_QA_mark2/helpdesk_qa_mark3.5.py



# Helpdesk QA Chatbot

This Streamlit application implements a simple Question-Answering (QA) chatbot for a helpdesk, leveraging LangChain, LangGraph, Google's Generative AI, and FAISS for efficient information retrieval. The bot is designed to understand user questions, find the most relevant answer from a predefined knowledge base, and provide a structured response.

## Key Features:

*   **Streamlit User Interface:** Provides an interactive chat interface for users to ask questions and receive answers.
*   **Google Generative AI Integration:** Utilizes `gemini-2.5-flash` for advanced natural language understanding, specifically to rephrase user questions into a canonical form suitable for retrieval.
*   **Retrieval-Augmented Generation (RAG):** Employs a FAISS vector store for fast and efficient similarity search over a knowledge base.
*   **FAISS Persistence:** The FAISS index is saved to disk (`faiss_index` directory) and reloaded on subsequent runs, speeding up application startup. It's only rebuilt if the index doesn't exist or if the cache is cleared.
*   **In-Memory Knowledge Base:** The helpdesk's QA data is defined directly within the script as a Python dictionary (`csv_data`), which is then converted into a `Helpdesk_bot_QA.csv` file for consistency before building the FAISS index.
*   **LangGraph State Machine:** Orchestrates the chatbot's workflow, managing the state between different processing steps (intent recognition, document retrieval, response generation).
*   **Relevance-Based Retrieval:** Implements a relevance threshold (set at 0.5) to filter out less relevant retrieved documents, ensuring the chatbot only provides confident answers. If no sufficiently relevant document is found, it informs the user.
*   **Structured Responses:** Formats the final answer clearly, including a summary of the agent's actions (original question, interpreted question, and retrieved document's question).
*   **Conversation Logging:** Automatically logs all chatbot responses to a timestamped file (`user_output_log_*.txt`) for review and analysis.

## Core Workflow:

1.  **Initialization:**
    *   Loads the Google API key from a `.env` file.
    *   Initializes Google Generative AI embeddings (`text-embedding-004`) and the `gemini-2.5-flash` model.
    *   Creates or loads the FAISS vector store: If the `faiss_index` directory exists, it loads the pre-built index; otherwise, it generates a `Helpdesk_bot_QA.csv` from internal data, builds a new FAISS index, and saves it.
    *   Compiles a LangGraph workflow defining the sequence of operations.
2.  **User Interaction:**
    *   Users input questions via the Streamlit chat interface.
    *   The user's question is added to the chat history.
3.  **Graph Execution (`intent_and_retrieve` node):**
    *   The `gemini-2.5-flash` model rephrases the user's question into a canonical FAQ format.
    *   A similarity search is performed against the FAISS vector store using the rephrased question.
    *   The most similar document is retrieved, and its relevance score is checked against a threshold.
4.  **Graph Execution (`generate_refined_response` node):**
    *   If a sufficiently relevant document is found, the answer is extracted and presented to the user along with a summary of the retrieval process.
    *   If no relevant document is found, the bot apologizes and suggests rephrasing the question.
5.  **Output:**
    *   The bot's response is displayed in the Streamlit chat and appended to the session's log file.

---
