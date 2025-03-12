from typing import Literal, List, Optional, Union
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from utils import *
from pydantic import BaseModel

load_dotenv()


# -----------------------------
# Define a Custom State Model (without retriever)
# -----------------------------
class CustomMessagesState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = []
    context: Optional[str] = None
    binary_score: Optional[str] = None
    response: Optional[str] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return hasattr(self, item)

    def keys(self):
        return self.__dict__.keys()

# -----------------------------
# Global Initialization (Define retriever once)
# -----------------------------
documents = load_documents()  # Load all document chunks
retriever = load_embeddings(documents, "initial query")  # Create retriever once

# -----------------------------
# Helper function to extract latest human query
# -----------------------------
def get_latest_query(state: CustomMessagesState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

# -----------------------------
# Define the Graph Nodes
# -----------------------------
def retrieve_node(state: CustomMessagesState) -> CustomMessagesState:
    """
    Retrieve top relevant document chunk using the global retriever.
    """
    query = get_latest_query(state)
    doc = top_chunk(retriever, query)  # Fetch top document chunk (a Document object)
    # Convert document to string (using its attribute if available)
    context = doc.page_content if hasattr(doc, "page_content") else str(doc)
    state["context"] = context
    state.messages.append(AIMessage(content="retrieve node: context retrieved."))
    return state

def grade_node(state: CustomMessagesState) -> CustomMessagesState:
    """
    Assess whether the retrieved context is relevant to the query.
    """
    query = get_latest_query(state)
    context = state.context
    binary_score = assess_retrieve_docs(query, context)
    state["binary_score"] = binary_score
    state.messages.append( AIMessage(content=f"grade node: graded with score {binary_score}."))
    return state

def check_relevance(state: CustomMessagesState) -> Literal["web_search", "generate"]:
    """
    If the retrieved document is irrelevant, perform a web search.
    """
    return "web_search" if state.binary_score == "خیر" else "generate"

def web_search_node(state: CustomMessagesState) -> CustomMessagesState:
    """
    Perform web search if retrieval fails.
    """
    query = get_latest_query(state)
    doc = search_web(query)
    # Convert document to string (using its attribute if available)
    context = doc.page_content if hasattr(doc, "page_content") else str(doc)
    state["context"] = context
    state.messages.append(AIMessage(content="web_search node: performed web search."))
    return state

def generate_node(state: CustomMessagesState) -> CustomMessagesState:
    """
    Generate the final response using the retrieved context.
    """
    query = get_latest_query(state)
    response = generate_response(retriever, query, documents)  # Use global retriever
    state["response"] = response
    state.messages.append(AIMessage(content=response))
    state.messages.append( AIMessage(content="generate node: final response generated."))
    return state

# -----------------------------
# Build the State Graph
# -----------------------------
workflow = StateGraph(CustomMessagesState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", check_relevance)
workflow.add_edge("web_search", "generate")

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# -----------------------------
# Run the Pipeline
# -----------------------------
def query_pipeline(query: str) -> str:
    initial_state = CustomMessagesState(messages=[HumanMessage(content=query)])
    
    # Ensure proper attribute access
    final_state = app.invoke(initial_state, config={"configurable": {"thread_id": 42}})
    
    return final_state.get("response", "No response generated.")  # <-- Fix

if __name__ == "__main__":
    q = 'تفسیر بیت زیر چیست: /"یاد من کن پیش تخت آن عزیز / تا مرا هم واخرد زین حبس نیز"'
    result = query_pipeline(q)
    print(result)
