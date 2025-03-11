import os
import stat
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from colorama import Fore
from langchain_openai import ChatOpenAI
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGUAGE_MODEL = "gpt-4 turbo"
llm = ChatOpenAI()


#### ارزیاب بازیابی : ارزیاب ارتباط ####
class GradeDocuments(BaseModel):
    """امتیاز باینری برای ارزیابی ارتباط اسناد بازیابی‌شده."""

    binary_score: str = Field(description="اسناد نسبت به پرسش کاربر مرتبط هستند: 'بله' یا 'خیر'")

    def get_score(self) -> str:
        """امتیاز باینری را به صورت رشته برمی‌گرداند."""
        return self.binary_score


def get_score(self) -> str:
    """امتیاز باینری را به صورت رشته برمی‌گرداند."""
    return self.binary_score

# استفاده از LLM با فراخوانی تابع ساختارمند
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt 

system_template = """تو یک ارزیاب هستی که ارتباط اسناد بازیابی شده زیر:/
{documents}/
با پرسش کاربر:/
{question}/
را تعیین می کنید. اگر سند شامل کلمه های کلیدی با معانی مرتبط با پرسش باشد آنرا به عنوان مرتبط علامتگداری کرده و پاسخ:
بله/
و اگر نه پاسخ:
خیر/
را انتخاب کنید./
سعی کن در پاسخ خود سخت گیر باشی و اگر اطمینان نداری که سند مرتبط است پاسخ خیر را انتخاب کنی./"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["documents", "question"],
    template="{question}",
)
grader_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def load_documents():
    """
    Load all text files from a directory, split them into chunks,
    and add metadata with 'doc_id' and 'chunk_index' for each chunk.
    """
    loader = DirectoryLoader("./output_text/", glob="*.txt")  # Load all .txt files
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    for raw_doc in raw_documents:
        # Get a document identifier. Here we use the 'source' metadata if available.
        doc_id = raw_doc.metadata.get("source", "unknown")
        chunks = text_splitter.split_text(raw_doc.page_content)
        for idx, chunk in enumerate(chunks):
            new_doc = Document(page_content=chunk, metadata={"doc_id": doc_id, "chunk_index": idx})
            all_chunks.append(new_doc)
    return all_chunks


def load_embeddings(documents, user_query):
    """
    Create or load a Chroma vector store from a set of documents.
    """
    persist_directory = './chroma_cache'  # Directory to store embeddings
    embedding_model = OpenAIEmbeddings()

    # Ensure the directory exists and has write permissions
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    else:
        if not os.access(persist_directory, os.W_OK):
            print(f"Error: No write access to {persist_directory}. Fixing permissions...")
            try:
                os.chmod(persist_directory, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            except Exception as e:
                print(f"Failed to change directory permissions: {e}")
                return None

    try:
        # Load or create Chroma vector store
        if not os.listdir(persist_directory):  # Empty directory means no existing DB
            #print("Initializing new ChromaDB instance...")
            db = Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)
            db.persist()
        else:
            print("Loading existing ChromaDB instance...")
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

        # For debugging: perform a similarity search with score and print the top result
        docs_with_scores = db.similarity_search_with_score(user_query, k=1)
        if docs_with_scores:
            top_doc, score = docs_with_scores[0]
            #print("\nRetrieved Document (for debugging):\n")
            #print(format_docs([top_doc]))
            #print("\nSimilarity Score:", score)
        else:
            print("No documents retrieved for the query.")

        return db.as_retriever()

    except Exception as e:
        print(f"Error while loading ChromaDB: {e}")
        return None
    
def top_chunk(retriever , query):

    retrieved_chunks = retriever.get_relevant_documents(query)
    if not retrieved_chunks:
        return "No relevant document found."

    # Retrieve the top (most relevant) chunk and extract doc_id
    top_chunk = retrieved_chunks[0]

    return top_chunk.page_content

def assess_retrieve_docs(query, context):

    retrieval_grader = grader_prompt | structured_llm_grader | get_score
    binary_score = retrieval_grader.invoke({"question": query, "documents": context})
    
    return binary_score 

template: str = """/
    فرض کن تو یک مفسر مثنوی هستی. جواب سوال زیر را بده: /
      {question} /
   از محتوای زیر برای پیدا کردن مفاهیم و زمینه مربوطه استفاده کن. محتوای زیر از جلسات تفسیر مثنوی معنوی عبدالکریم سروش گرفته شده است./
      {context} /
      این محتوا از جلسه شماره زیر گرفته شده است:/
      {doc_name}/
       . سعی کن از این محتوا برای فهمیدن داستان و تاریخ مربوطه و ابیات مجاور بیت مورد سوال استفاده کنی./
       در پاسخی که میدهی سعی کن به زمینه داستانی و تاریخی و اشعار مجاور شعر مورد سوال در مثنوی اشاره کنی./
       در پاسخ خود به شماره جلسه ای که در مورد این شعر صحبت شده نیز اشاره کن./
       در پاسخ خود به تمام جزییاتی که از متن جلسه دریافت می کنی و مرتبط با سوال مطرح شده است اشاره کن.
    """

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}",
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

### Question Re-writer - Knowledge Refinement ####
# Prompt 
prompt_template = """با در نظر گرفتن سوال زیر:/
{question},/
 دقت کن که سوال را به گونه ای بازنویسی کنی که مدل زبانی بتواند بهترین پاسخ را ارائه دهد./
 به سوال پاسخ نده. فقط سوال را بازنویسی کن./"""

system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
human_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question"],
    template="{question}",
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt]
)

### Web Search Tool - Knowledge Searching ####
web_search_tool = TavilySearchResults(k=3) 


def rewrite_query(query):
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": query})

def search_web(query):
    docs = web_search_tool.invoke({"query": query})
    web_results = "\n".join([d["content"] for d in docs])
    return Document(page_content=web_results)
    
def generate_response(retriever, query, documents):
    retrieved_chunks = retriever.get_relevant_documents(query)
    if not retrieved_chunks:
        return "No relevant document found."

    # Retrieve the top (most relevant) chunk and extract doc_id
    top_chunk = retrieved_chunks[0]
    doc_id = top_chunk.metadata.get("doc_id")
    chunk_index = top_chunk.metadata.get("chunk_index")

    # Find all chunks from the same document
    same_doc_chunks = [doc for doc in documents if doc.metadata.get("doc_id") == doc_id]
    same_doc_chunks = sorted(same_doc_chunks, key=lambda d: d.metadata.get("chunk_index", 0))

    # Define a window: e.g. 20 chunks before and after the top chunk
    start = max(0, chunk_index - 20)
    end = min(len(same_doc_chunks), chunk_index + 20)
    aggregated_context = "\n\n".join([doc.page_content for doc in same_doc_chunks[start:end]])

    # Build the chain and invoke it with the additional 'doc_name' variable
    chain = chat_prompt_template | llm | StrOutputParser()
    input_vars = {"context": aggregated_context, "question": query, "doc_name": doc_id}
    return chain.invoke(input_vars)

