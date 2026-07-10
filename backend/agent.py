#-------------------------------------------------------------------------------
#                               Import Statements
#-------------------------------------------------------------------------------
import os 
from config import *
from typing import TypedDict,List,Literal,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolCall
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langchain_core.tools import tool 
from vectorstore import get_retriever
from langchain_tavily import TavilySearch

# Setting up the environment key
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
tavily = TavilySearch(max_result = 3 , topic = "general")

#-------------------------------------------------------------------------------
#                               Schemas and State Define
#-------------------------------------------------------------------------------

# Pydantic Schema
class RouteDecision(BaseModel):
    route : Literal['rag' , 'web' , 'answer' , 'end']
    reply : str | None = Field(None , description = "Filled only when route == 'end' ")

class RagJudge(BaseModel):
    sufficient : bool = Field(... , description = "True if retrieved informationis sufficient to answer the user query and False otherwise")


# LLM Instances with structured schemas
router_llm = ChatGroq(model = "llama-3.3-70b-versatile" , temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model = "llama-3.3-70b-versatile" , temperature=0).with_structured_output(RagJudge)
answer_llm = ChatGroq(model = "llama-3.3-70b-versatile" , temperature=0.7)


# State Defination
class AgentState(TypedDict , total = False):
    messages : List[BaseMessage]
    route : Literal['rag' , 'web' , 'answer' , 'end']
    rag : str 
    web : str 
    web_search_enable : bool 

#-------------------------------------------------------------------------------
#                               Tools Defination
#-------------------------------------------------------------------------------

@tool
def rag_search_tool(query : str) -> str:
    """
        Retrieving the top k chunks from the knowledgebase and empty string if none
        Args:
            query(str) : User question
        Returns:
            Either string of documents or empty string
    """
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query , k = 3)
        return "\n\n".join(d.page_content for d in docs) if docs else ""

    except Exception as e:
        return f"RAG Error : {str(e)}"


@tool
def web_search_tool(query : str) -> str:
    """
        Retriving the information from the Tavily api 
        Args:
            query(str) : User question
        Returns:
            Either string of documents or empty string
    """
    try:
        result = tavily.invoke({"query" : query})
        if isinstance(result , dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title' , 'No title')
                content = item.get('content' , 'No Content')
                url = item.get('url' , '')
                formatted_results.append(f"Title : {title}\nContent : {content}\nURL : {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No Results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR : {str(e)}"
    

#-------------------------------------------------------------------------------
#                               Node Definations
#-------------------------------------------------------------------------------

def router_node(state : AgentState) -> AgentState:
    """
        For making decision to which path to choose based on the
        user query. It will check with llm and decide the next route
        Args:
            state(AgentState) : langgraph agent for information passing
    """
    print("Entering the router node...")

    # Extract query -> Check on the messages of the state and reversed it first and check if there is any humanmessage or not
    query = next(
        (
            m.content for m in reversed(state['messages'])
            if isinstance(m,HumanMessage)
        ),
        ""
    )
    # Get WebSearch info
    web_search_enabled = state.get("web_search_enable" , True)
    print(f"Router received web search info : {web_search_enabled}")

    # Overall System prompt 
    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )

    # Append more instruction into system prompt according to the websearch allowed or not
    if web_search_enabled:
        # If websearch allowed then show for what kind of queries go to web and for what query go to rag
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        # If websearch not allowed then show for what kind of queries go to web and for what query go to rag
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    # Now making the final append to the system prompt with all the available route above there is no end or answer mentioned well in both
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

    messages = [
        ('system' , system_prompt) , 
        ('user' , query) 
    ]

    # Invoking the llm 
    result : RouteDecision = router_llm.invoke(messages)
    initial_router_decision = result.route
    router_overrider_reason = None 
    
    # Override decision of router to go on web 
    if not web_search_enabled and result.route == "web":
        result.route = "rag" 
        router_overrider_reason = "Web search is disabled by user, Redirected to RAG"
        print(f"Router decision overriden  : change from Web to RAG")

    print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")

    # Setup the variable for return
    out = {
        "messages" : state['messages'] ,
        "route" : result.route,
        "web_search_enable" : web_search_enabled
    }

    # what if router need to override
    if router_overrider_reason:
        out['initial_router_decision'] = initial_router_decision
        out['router_override_reason'] = router_overrider_reason

    # What if route is going for end -> we have one reply in the pydantic model special for this only...
    if result.route == 'end':
        out['messages'] = state['messages'] + [AIMessage(content = result.reply or "Hello")]

    print("Exiting router node...")
    
    # Returning the output dict
    return out


#****************************************************************************************************************************************
#****************************************************************************************************************************************


def rag_node(state : AgentState) -> AgentState:
    """
        This node will call the retriever and fetch the document from the Pinecone
        Args:
            state(AgentState) : langgraph agent for information passing
    """
    print("Entering the rag_node")

    # Extract query -> Check on the messages of the state and reversed it first and check if there is any humanmessage or not
    query = next(
        (
            m.content for m in reversed(state['messages'])
            if isinstance(m,HumanMessage)
        ),
        ""
    )

    # Get WebSearch info
    web_search_enabled = state.get("web_search_enable" , True)
    print(f"Router received web search info : {web_search_enabled}")
    print(f"RAG Query : {query}")

    # Getting chunks
    chunks : str = rag_search_tool.invoke(query)

    # Logic to handle chunks
    if chunks.startswith("RAG Error"):
        print(f"RAG Error : {chunks} , checing with web search enabled status")
        # if rag fails check websearch status
        next_route = "web" if web_search_enabled else "answer"
        return {**state , "rag" : "" , "route" : next_route}
    
    if chunks:
        print(f"Retrived RAG Chunks : {chunks[:500]}...")
    else:
        print("No RAG chunks retrieved")

    # Check the information is relevant or not
    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "Consider if the retrieved text directly addresses the question's core and provides enough detail."
            "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient."
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
            "\n\nExample 1: Question: 'What is the capital of France?' Retrieved: 'Paris is the capital of France.' -> {\"sufficient\": true}"
            "\nExample 2: Question: 'What are the symptoms of diabetes?' Retrieved: 'Diabetes is a chronic condition.' -> {\"sufficient\": false} (Doesn't answer symptoms)"
            "\nExample 3: Question: 'How to fix error X in software Y?' Retrieved: 'No relevant information found.' -> {\"sufficient\": false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
    ]

    verdict : RagJudge = judge_llm.invoke(judge_messages)
    print(f"RAG Judge verdict : {verdict.sufficient}")
    print("Exiting rag_node")

    # Decide the next route based on sufficiency and web_search information
    if verdict.sufficient:
        # If sufficient information retrieved then go to answer
        next_route = "answer"
    else:
        # If not sufficient info retrieved then check if the web_search enabled or not and if yes then go to web and if not then go to answer
        next_route = "web" if web_search_enabled else "answer"
        print(f"Retrieved Documents are not sufficient, Webs search enabled : {web_search_enabled}. So next route : {next_route}")

    return {
        **state , "rag" : chunks , "route" : next_route , "web_search_enable" : web_search_enabled
    }


#****************************************************************************************************************************************
#****************************************************************************************************************************************

    
def web_node(state :AgentState) -> AgentState:
    """
        Node for Tavily search and retriving the data using that tavily API
        Args:
            state(AgentState) : langgraph agent for information passing
    """
    print("Entering WEB Node...")

    # Extract query -> Check on the messages of the state and reversed it first and check if there is any humanmessage or not
    query = next(
        (
            m.content for m in reversed(state['messages'])
            if isinstance(m,HumanMessage)
        ),
        ""
    )

    # Get WebSearch info
    web_search_enabled = state.get("web_search_enable" , True)
    print(f"Router received web search info : {web_search_enabled}")

    if not web_search_enabled:
        print("Web Search node entered but Web search disabled")
        return {**state , "web" : "Web search was disabled by user" , "route" : "answer"}

    print(f"Web search query : {query}")
    snippets : str =  web_search_tool.invoke({'query' : query})

    # If Error in the websearch
    if snippets.startswith("WEB_ERROR"):
        print(f"Web error : {snippets}. Predicting to answer with limited info")
        return {**state , "web" : "" , "route" : "answer"}
    
    # If no issue in Websearch
    print(f"Web Snippets retrieved : {snippets[:200]}...")
    print("Exiting from web_node")

    return {
        **state , "web" : snippets , "route" : "answer"
    }


#****************************************************************************************************************************************
#****************************************************************************************************************************************


def answer_node(state : AgentState) -> AgentState:
    """ 
        Synthesizes all gathered information and generates the final
        answer to the user's query. IT will get info from RAD and / or web
        Args:
            state(AgentState) : langgraph agent for information passing
    """
    print("Entering answer node...")

    # Extract query -> Check on the messages of the state and reversed it first and check if there is any humanmessage or not
    user_query = next(
        (
            m.content for m in reversed(state['messages'])
            if isinstance(m,HumanMessage)
        ),
        ""
    )

    # providing context
    ctx_parts = [] 
    if state.get("rag"):
        ctx_parts.append("knowledge Base Information : \n" + state['rag'])
    if state.get("web"):
        if state['web'] and not state['web'].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results : \n" + state['web'])

    context = "\n\n".join(ctx_parts)

    if not context.strip():
        context = "No external context is available for this query, Try to answer based on general knowledge"

    prompt = f"""Please answer the user's question using the provided context.
                If the context is empty or irrelevant, try to answer based on your general knowledge.

                Question: {user_query}

                Context:
                {context}

                Provide a helpful, accurate, and concise response based on the available information."""

    print(f"Prompt sent to the answer_llm : {prompt[:500]}")
    ans = answer_llm.invoke([HumanMessage(content = prompt)]).content 
    print(f"Final answer : {ans[:200]}...")
    print("Exiting answer_node")
    return {
        **state , 
        "messages" : state['messages'] + [AIMessage(content = ans)]
    }


#-------------------------------------------------------------------------------
#                              Routing Helpers
#-------------------------------------------------------------------------------


def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_) -> Literal["answer"]:
    return "answer"


#-------------------------------------------------------------------------------
#                              Graph Building
#-------------------------------------------------------------------------------

def build_agent():
    """
        Building graph, Adding nodes, defining edges and compiling it
    """
    graph = StateGraph(AgentState)
    
    # Adding nodes to graph
    graph.add_node("router",router_node)
    graph.add_node("rag_lookup",rag_node)
    graph.add_node("web_search",web_node)
    graph.add_node("answer" , answer_node)

    # creating edges
    graph.set_entry_point("router")
    graph.add_conditional_edges("router" , from_router , {
        "rag" : "rag_lookup",
        "web" : "web_search",
        "answer" : "answer",
        "end" : END 
    })
    graph.add_conditional_edges("rag_lookup" , after_rag , {
        "web" : "web_search",
        "answer" : "answer"
    })
    graph.add_conditional_edges("web_search" , after_web , {
        "answer" : "answer"
    })
    graph.add_edge("answer" , END)

    agent = graph.compile(checkpointer=MemorySaver())
    return agent 

rag_agent = build_agent()