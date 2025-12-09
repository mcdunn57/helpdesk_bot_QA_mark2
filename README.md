# Helpdesk QA Bot mk2

# Gemini-Powered RAG Helpdesk Chatbot

## Overview

This Python script implements a sophisticated, yet easy-to-understand, Helpdesk Q&A chatbot. It uses a Retrieval-Augmented Generation (RAG) architecture powered by Google's Gemini models and the LangGraph framework.

The primary goal is to provide accurate answers to user questions by first finding the most relevant information in a knowledge base and then generating a clear, helpful response. It is designed to be robust, handling irrelevant questions gracefully and providing users with transparency into its reasoning process.

## Key Features

*   **RAG Architecture**: The bot retrieves relevant documents from a knowledge base before generating an answer, ensuring responses are grounded in factual data.
*   **Intent Rephrasing**: It uses a Gemini model to interpret the user's raw input and rephrase it into a "canonical" question. This improves the accuracy of the document retrieval step.
*   **Vector Store with Persistence**: It uses a FAISS vector store to index the knowledge base for efficient similarity searches. The script automatically saves the created index to disk (`faiss_index/`) and loads it on subsequent runs, avoiding the need to re-process the source documents every time.
*   **Relevance Scoring**: After retrieving a document, the bot checks its relevance score. If the score is below a set threshold (0.6), the document is discarded, allowing the bot to intelligently respond that it doesn't have a relevant answer.
*   **LangGraph State Machine**: The agent's logic is structured as a graph using LangGraph. This makes the flow of control explicit and easy to follow, from intent recognition to final response generation. The graph consists of two main nodes:
    1.  `intent_and_retrieve`: Rephrases the question and retrieves a relevant document.
    2.  `generate_refined_response`: Creates the final answer for the user, including a summary of the agent's actions for transparency.
*   **Transparent Responses**: The final output not only provides the answer but also shows the user their original question, how the bot interpreted it, and which document it used, building user trust.
*   **Conversation Logging**: All interactions are automatically saved to a timestamped log file (e.g., `user_output_log_20240705_103000.txt`) for later analysis.

## How It Works

1.  **Setup**: The script loads a Google API key from a `.env` file and initializes the Gemini LLM and embedding models.
2.  **Vector Store Initialization**: It checks for a local FAISS index. If one exists, it's loaded. If not, it reads a `Helpdesk_bot_QA.csv` file, creates embeddings for the Q&A pairs, and saves the new index to disk.
3.  **User Input**: The bot waits for a user to ask a question in the command line.
4.  **Intent & Retrieval**: The user's question is sent to the `intent_and_retrieve` node. A Gemini model rephrases the question into a canonical form. This rephrased question is used to search the FAISS vector store.
5.  **Relevance Check**: The top-ranked document is checked against a relevance score threshold.
6.  **Response Generation**: The `generate_refined_response` node takes the retrieved document (if any) and crafts the final answer. If no relevant document was found, it informs the user. Otherwise, it presents the answer along with a summary of its internal steps.
7.  **Output & Logging**: The final response is printed to the console and appended to the log file.

