# main.py
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# -------- STEP 1: Load API Keys --------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
NEWS_KEY = os.getenv("NEWS_API_KEY")

# -------- STEP 2: FastAPI App --------
app = FastAPI(title="LangGraph Multi-Agent API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OpenAI LLM (for RAG Agent)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY)

# -------- STEP 3: Load PDF and Create Vectorstore --------
def load_pdf_to_vectorstore(pdf_path="data/Mfarooq AI_Engineer CV.pdf"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_pdf_to_vectorstore()

# -------- WEATHER AGENT --------
def weather_agent(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric"
    res = requests.get(url).json()
    if res.get("main"):
        temp = res["main"]["temp"]
        desc = res["weather"][0]["description"]
        return f"Weather in {city}: {temp}°C, {desc}"
    return "❌ Weather info not available."

# -------- NEWS AGENT --------
def news_agent():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_KEY}"
    res = requests.get(url).json()
    if "articles" in res:
        headlines = [art["title"] for art in res["articles"][:5]]
        return "Top News:\n" + "\n".join([f"- {h}" for h in headlines])
    return "❌ News not available."

# -------- RAG AGENT (PDF + LLM) --------
def rag_agent(query: str):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    return f"Answer: {result['result']}"

# -------- ROUTER FUNCTION --------
def router_agent(query: str):
    q_lower = query.lower()
    if "weather" in q_lower or "temperature" in q_lower:
        city = query.split("in")[-1].strip() if "in" in q_lower else "place"
        return weather_agent(city)
    elif "news" in q_lower or "headline" in q_lower:
        return news_agent()
    else:
        return rag_agent(query)

# -------- FASTAPI ENDPOINT --------
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(req: QueryRequest):
    result = router_agent(req.query)
    return {"query": req.query, "response": result}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
