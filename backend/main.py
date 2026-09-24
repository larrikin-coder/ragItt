import os
import time
from typing import List,Dict,Any
import tempfile
from fastapi import FastAPI,HTTPException,status,UploadFile,File
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader

from agent import rag_agent
from vectorStore import add_documents

app = FastAPI(title="LangGraph RAG Agent API",description="API for the the Langgraph-powered RAG Agent with Pinecone and groq")
memory = MemorySaver()


class TraceEvent(BaseModel):
    step:int
    node_name:str
    description:str
    details:Dict[str,Any] = Field(default_factory=dict)
    event_type:str
    
class QueryRequest(BaseModel):
    session_id:str
    query:str
    enable_web_search:bool =True

class AgentResponse(BaseModel):
    response:str
    trace_event:List[TraceEvent] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int    
    
    
@app.post('/upload-document',response_model=DocumentUploadResponse,status_code=status.HTTP_200_OK)
async def upload_document(file:UploadFile = File(...)):
    """
    Uploads PDF file extracts the text and adds it to the RAG Knowledge Base
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
        file_content = await file.read()
        tmp_file.write(file_content)
        temp_file_path = temp_file.name
    
    print(f"Recieved pdf for upload: {file.filename}. Save temporarily to {temp_file_path}")
    
    
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        total_chunks_added = 0
        if document:
            full_text_context =  "\n\n".join([doc.page_content for doc in  documents])
            add_documents(full_text_context)
            total_chunks_added = len(documents)
        return DocumentUploadResponse(
            message = f"PDF {file.filename} successfully uploaded and indexed",
            filename = file.filename,
            processed_chunks = total_chunks_added 
            
        )
        
    except Exception as e:
        print(f"Error processing PDF  document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {e}"
        )
        
    finally:
        if os.path.exist(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

@app.post("/chat/",response_model=AgentResponse)
async def chat_with_agent(request: QueryRequest):
    trace_events_for_frontend: List[TraceEvent] = []
    
    try:
        config = {
            "configurable":{
                "thread_id": request.session_id,
                "web_search_enabled":  request.enable_web_search
            }
        }
        inputs = {"messages": [HumanMessage(content=request.query)]}
        
        final_message = ""
        print(f"Starting Agent Stream for session {request.session_id}")
        print(f"Web search Enabled : {request.web_search}")
        
        for i,s in enumerate(rag_agent.stream(inputs,config=config)):
            current_node_name = None
            node_output_state = None
            if '__end__' in s:
                current_node_name = None
                node_output_state = s['__end__']
            else:
                current_node_name = list(s.keys())[0]
                node_output_state = s[current_node_name]
                
            event_description =  f"Executing node: {current_node_name}"
            event_details = {}
            event_type = "generic_node_execution"

            if current_node_name == "router":
                route_decision = node_output_state.get('route')
                initial_decision = node_output_state. get('initial_router_decision',route_decision)
                override_reason = node_output_state.get('router_override_reason',None)
                
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_details = f"Error during agent invocation {e}"
        print(error_details)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"Internal Server Error: {e}")
    
