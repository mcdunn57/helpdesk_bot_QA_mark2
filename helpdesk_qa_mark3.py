# --- 1. Installation ---
# Run this cell first to install the required libraries.
# !pip install langchain langgraph langchain_google_genai pandas faiss-cpu google-generativeai python-dotenv

# --- 2. Imports and Setup ---
import os
import operator
import getpass
import pandas as pd
import streamlit as st
import json
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import langchain_google_genai
import langchain_core
import langgraph
import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime

# --- 3. API Key and LLM Initialization ---
# Ensure you have a .env file in the same directory with the line:
# GOOGLE_API_KEY="your-actual-api-key"
load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in the .env file or environment variables. Please create a .env file and add your key (e.g., GOOGLE_API_KEY='your-key-here').")
    
    # The API key is already set as an environment variable by load_dotenv(),
    # but we also need to configure the native genai library.
    genai.configure(api_key=api_key)

except ValueError as e:
    print(f"ERROR: {e}")
    # Stop execution if the key is not found
    raise e

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# This is the native Google library model, which we will use for direct, robust calls.
native_model = genai.GenerativeModel('gemini-2.5-flash')



# --- 4. Persist and Load Vector Store ---
@st.cache_resource
def initialize_vector_store():
    """Initializes the FAISS vector store from in-memory data."""
    print("Creating a new FAISS index from in-memory data...")
    
    # Define the data for the helpdesk bot
    csv_data = {
        'Question': [
            "How do I reset my password?",
            "What are the system requirements for the software?",
            "How can I contact customer support?",
            "What is the return policy?",
            "Where can I find the user manual?"
        ],
        'Answer': [
            "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
            "The software requires Windows 10 or later, 8GB of RAM, and at least 5GB of free disk space.",
            "You can contact customer support by emailing help@example.com or by calling our hotline at 1-800-555-1234 during business hours.",
            "Our return policy allows for returns within 30 days of purchase, provided the product is in its original condition.",
            "The user manual is available for download in the 'Help' section of our website."
        ]
    }
    df_raw = pd.DataFrame(csv_data)
    
    # Optionally save to CSV if you want the file to exist, but the vector store is built from df_raw
    df_raw.to_csv('Helpdesk_bot_QA.csv', index=False)
    print("Created 'Helpdesk_bot_QA.csv' with sample data.")

    # Convert DataFrame rows to Document objects
    documents = [Document(page_content=f"Question: {row['Question']}\nAnswer: {row['Answer']}") for index, row in df_raw.iterrows()]

    # Create the FAISS vector store directly from the documents
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"✅ New FAISS index created in memory.")
    return vector_store

# Initialize the vector store using the cached function
vector_store = initialize_vector_store()



# --- 5. Define the Graph State ---
# FIX: Updated state to pass the rephrased_question to the new response generator node.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    question: str
    rephrased_question: str
    retrieved_document: Document

# --- Helper function for logging ---
def write_to_log(log_filename, response_content):
    """Appends a response to the specified log file."""
    with open(log_filename, 'a', encoding='utf-8') as f:
        response_tuple = (response_content,)
        f.write(str(response_tuple) + '\n')
# --- 6. Define the Graph Nodes ---
def intent_and_retrieve(state: AgentState):
    """
    Node to interpret user intent, retrieve a document, and pass all info to the state.
    """
    print("--- 🤔 Executing node: intent_and_retrieve ---")
    question = state['question']
    
    # Create the prompt for the native LLM call. note: still not sure how category gets created - mcd
    template = """Rephrase the user's question into a canonical question suitable for an FAQ document.
You must respond with a JSON object. The most important key in this object should contain the rephrased question.

Example:
User question: I forgot my password, what do I do?
JSON response: {{"rephrased_question": "How do I reset my password?", "category": "account_management"}}

User question: {question}
"""
    prompt_text = template.format(question=question)

    # Invoke the LLM using the native google-generativeai library to bypass LangChain errors
    response = native_model.generate_content(prompt_text)
    raw_response_text = response.text
    print(f"    Raw response from LLM: {raw_response_text}")

    # Parse the JSON response
    if '```json' in raw_response_text:
        json_str = raw_response_text.split('```json')[1].split('```')[0].strip()
    else:
        json_str = raw_response_text
    response_data = json.loads(json_str)
    
    # Dynamically find the key containing the rephrased question
    rephrased_question = None
    for key in response_data.keys():
        if "question" in key.lower():
            rephrased_question = response_data[key]
            break
    if not rephrased_question:
        rephrased_question = list(response_data.values())[0]

    print(f"    Rephrased question for retrieval: '{rephrased_question}'")
    
    # FIX: Retrieve document with relevance score to handle irrelevant questions.
    # We call the vector store directly to get similarity scores.
    #note: phenomenon is that exact Q statements, even as rephrased Q still, will not get 100 similarity scores - mcd
    docs_and_scores = vector_store.similarity_search_with_relevance_scores(rephrased_question, k=1)

    retrieved_document = None
    if docs_and_scores:
        doc, score = docs_and_scores[0]
        print(f"    Top document retrieved with relevance score: {score:.4f}")
        # Set a threshold for relevance. Scores are 0-1, higher is better.
        #lowered this down to .6 after testing against the highly related test cases. didn't go below this though; allows less false negatives while 
        #still not seeing any false positives
        
        if score > 0.5:  # Adjust this threshold as needed
            retrieved_document = doc
            print("    ✅ Match is relevant enough.")
        else:
            print("    ❌ Match is NOT relevant enough. Discarding.")
    else:
        print("    No document found in vector store.")
    
    return {
        "retrieved_document": retrieved_document,
        "rephrased_question": rephrased_question,
        "question": question
    }

# FIX: Added a new node to generate the final, refined response.
def generate_refined_response(state: AgentState):
    """
    Node to generate a final, structured response for the user based on the retrieved document.
    """
    print("--- ✍️ Executing node: generate_refined_response ---")
    question = state["question"]
    rephrased_question = state["rephrased_question"]
    doc = state['retrieved_document']
    
    # This 'if' statement now correctly handles cases where the retriever
    # found a document but it was below the relevance threshold.
    if not doc:
        response_content = "I'm sorry, I couldn't find a relevant answer in our documentation. Please try rephrasing your question."
        return {"messages": [AIMessage(content=response_content)]}

    # Parse the Question and Answer from the document's content
    content = doc.page_content
    retrieved_q = content.split('Answer:')[0].replace('Question:', '').strip()
    retrieved_a = content.split('Answer:')[1].strip()

    # Create the final structured response as requested
    response_content = (
        f"{retrieved_a}\n\n"
        f"---\n"
        f"**Summary of Agent's Actions:**\n"
        f"* **Your Question:** \"{question}\"\n"
        f"* **Interpreted as:** \"{rephrased_question}\"\n"
        f"* **Retrieved Document:** \"{retrieved_q}\""
    )
    
    return {"messages": [AIMessage(content=response_content)]}


# --- 7. Build the Graph ---
@st.cache_resource
def get_graph():
    """Compiles and returns the LangGraph graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("intent_and_retriever", intent_and_retrieve)
    workflow.add_node("generate_refined_response", generate_refined_response)

    workflow.set_entry_point("intent_and_retriever")
    workflow.add_edge("intent_and_retriever", "generate_refined_response")
    workflow.add_edge("generate_refined_response", END)

    return workflow.compile()

# --- 8. Streamlit App ---
def main():
    st.title("📄 Helpdesk QA Chatbot")

    # Initialize log file in session state
    if "log_filename" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.log_filename = f"user_output_log_{timestamp}.txt"
        print(f"Logging chatbot responses to: {st.session_state.log_filename}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about our services"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                graph = get_graph()
                initial_input = {"question": prompt, "messages": [HumanMessage(content=prompt)]}
                final_state = graph.invoke(initial_input)
                response = final_state['messages'][-1].content
                st.markdown(response)
                # Log the response
                write_to_log(st.session_state.log_filename, response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
