from typing import TypeDict,List,Literal
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
import config


#Pydantic schema for structured output
class RouteDecision(BaseModel):
    route: Literal["rag","web","answer","end"]
    reply: str | None = Field(None,description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient:bool = Field(...,description="True if retrieved information is sufficient to answer the users question,False otherwise.")

#LLm instances with structured schemas
router_llm = ChatGroq(model="llama-3.1-8b-instant",temperature="0").with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama-3.1-8b-instant",temperature="0").with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama-3.1-8b-instant",temperature="0.7")


#State : Shared Data Structure
class AgentState(TypeDict,total=False):
    messages:List[BaseMessage]
    route: Literal["rag","web","answer","end"]
    rag:str
    web:str
    web_search_enabled:bool
    
#Node : For individual functions
def router_node(state:AgentState)->AgentState:
    print('Entering router Node')
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),"")
    web_search_enabled = state.get("web_search_enabled",True)
    print(f"Router recieved enabled {web_search_enabled}")
    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )
    
    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )
    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
        "\n\nExample routing decisions:"
        "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
        "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
        "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'"
    )
    
    messages = [("system",system_prompt),
                ("user",query)]
    
    result : RouteDecision = router_llm.invoke(messages)
    initial_router_decision = result.route
    router_overrider_reason = None
    
    #Override the router decision to go for the web search
    if not web_search_enabled and result.route=="web":
        result.route = "rag"
        router_overrider_reason = "Web search disabled by user, redirected to rag"
        print(f"Router Decision overriden changed from 'web' to 'rag'")
        
    print(f"Router final Decision {result.route},reply(if 'end'):{result.reply}")
    
    
    out = {
        "messages": state['messages'],
        "router": result.route,
        "web_search_enabled": web_search_enabled
    }
    
    if router_override_reason: # Add override info for tracing
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]
    
    print("--- Exiting router_node ---")
    return out
        