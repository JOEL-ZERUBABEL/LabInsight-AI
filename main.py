from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage,AIMessage
from typing import TypedDict,Annotated,Literal
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.tools import tool
llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
model=SentenceTransformer('all-MiniLM-L6-v2')

#Agent 1 -> Extracts info and format it
class Agent1(TypedDict):
    report_text:str
    test_name:str
    value:float
    unit:str
    reference_range:str
    report_data:str

#HbA1c: 8.2% (Reference: 4–5.6%)

def agent1(data:str)->dict:

    format={"report_text":data,"test_name":"","value":0.0,
            "unit":"","reference_range":"","report_data":""}

    prompt = f"""
    You are a medical report information extraction agent.

    Extract information ONLY from the given report.

    Report:
    {data}

    Fill these fields:
    {format}

    Rules:
    - Do not invent information.
    - If information is missing, return "Not available".
    - Return only the required fields in JSON format.
    """

    agent1response=llm.with_structured_output(Agent1,method="json_mode").invoke(prompt)
    return agent1response

class Agent2:
    class Agent2Output(TypedDict):
        test_name:str
        value:float
        unit:str
        interpretation:str
    
    #AGENT 2 Get report from agent 1 -> analyze with RAG -> LLM report
    def retrivel(self,chunksize=500,overlap=100,response=""):
        textofmedoc=""
        chunks=[]

        #creating embedding vector of medoc from chunking
        with open("medoc.txt","r",encoding="utf-8") as f:
            textofmedoc=f.read()
            start=0
            while start<len(textofmedoc):
                end=start+chunksize
                chunk=textofmedoc[start:end]
                chunks.append(chunk)
                start+=(chunksize-overlap)
            embeddingofmedoc=model.encode(chunks)


        #creating embedding vector of response
        textofresponse=""
        textofresponse+=str(response)
        embeddingoftextofresponse=model.encode([textofresponse])

        #faiss creation
        dimension=embeddingofmedoc.shape[1]
        index=faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddingofmedoc))

        #retrivel
        dist,indices=index.search(embeddingoftextofresponse,k=2)

        return indices,chunks
            

    def llm(self,indices,chunks,response):   
        retrieved_chunks = []

        for i in indices[0]:
            retrieved_chunks.append(chunks[i])

        prompt = f"""
                You are RAG agent AGENT2 ,
                Agent 1 extracted this information:{response}

        Relevant medical knowledge retrieved from the
        medical knowledge base:
        {retrieved_chunks}

        Use the retrieved knowledge to interpret the
        Agent 1 result.

        Do not diagnose.
        Do not prescribe medication.
        Do not invent information.
        Return the required fields in JSON format."""
            
        agent2response=llm.with_structured_output(self.Agent2Output,method="json_mode").invoke(prompt)
        return agent2response

# Agent 3 -> Gets Agent 2 interpretation -> prioritizes the findings

from typing import TypedDict


class Agent3(TypedDict):
    test_name:str
    value:float
    unit:str
    interpretation:str
    priority:str
    reason:str


def priority(response: dict) -> dict:

    prompt = f"""
    You are Agent 3, a medical finding prioritization agent.

    Your job is to prioritize the findings that were already
    interpreted by Agent 2.

    Agent 2 result:
    {response}

    Rules:

    1. Use ONLY the information provided by Agent 2.
    2. Do not diagnose any disease.
    3. Do not prescribe or recommend medication.
    4. Do not invent symptoms, patient history, or missing information.
    5. Do not change the interpretation given by Agent 2.
    6. Decide the attention level for the finding:
       - HIGH
       - MEDIUM
       - LOW
    7. Give a short reason for the priority.
    8. If there is not enough information to determine priority,
       use "UNCERTAIN" and explain why.
    9.Return the required fields in JSON format.
    """

    agent3response = llm.with_structured_output(Agent3,method="json_mode").invoke(prompt)

    return agent3response


#Agent 4 Just Validate agent 3 and agent 2
class Agent4(TypedDict):
    valid: bool
    missing_fields: list[str]
    reason: str
                                             
def validate(agent1response:dict,agent3response:dict)->dict:
    prompt = f"""
    You are Agent 4.
    Your job is ONLY to validate the response of Agent 3
    using the response of Agent 1.
    Agent 1 response:
    {agent1response}
    Agent 3 response:
    {agent3response}

    Check these things:
    1. test_name in Agent 3 must match test_name from Agent 1.
    2. value in Agent 3 must match value from Agent 1.
    3. unit in Agent 3 must match unit from Agent 1.
    4. interpretation must be present.
    5. priority must be present.
    6. reason must be present.

    If all required fields are present and the Agent 3
    test_name, value, and unit match Agent 1:

    valid = True
    missing_fields = []
    reason = "Agent 3 response is complete and consistent with Agent 1."

    If something is missing or does not match:
    valid = False
    missing_fields = [list the missing or incorrect fields]
    reason = "Explain what is missing or inconsistent."

    Do not interpret the medical result.
    Do not change any values.
    Do not provide medical advice.
    Return the required fields in JSON format."""

    agent4response=llm.with_structured_output(Agent4,method="json_mode").invoke(prompt)
    return agent4response

'''class Agent5(TypedDict):
    summary: str
    findings: str
    priority: str
    next_steps: str
    disclaimer: str'''

class Agent5(TypedDict):
    summary: str
    explanation: str
    priority: str
    next_steps: str
    missing_info: list
    disclaimer: str


def agent5(report):

    prompt = f"""
    You are Agent 5, the final response agent.

    Your job is to create a clear and easy-to-understand
    response using ONLY the validated report information.

    Validated report:
    {report}

    Your response must:

    1. Give a short summary of the report.
    2. Explain the important findings in simple language.
    3. Mention the priority given by the previous agent.
    4. Give general next-step guidance based only on the
       information provided.
    5. Do not invent any patient information.
    6. Do not diagnose any disease.
    7. Do not prescribe or recommend medication.
    8. Do not change the values or interpretations provided
       by the previous agents.
    9. If information is missing, clearly mention it.
    10. End with an appropriate medical disclaimer.

    Make the response concise, clear, and understandable
    to a non-medical person.
    Return the required fields in JSON format.
    """

    response = llm.with_structured_output(Agent5,method="json_mode").invoke(prompt)

    return response

class State(TypedDict):
    report_text:str
    agent1response:dict
    indices:any
    chunks:list
    agent2response:dict
    agent3response:dict
    agent4response:dict
    agent5response:dict

# Agent 1 node
def agent1_node(state:State):
    response=agent1(state["report_text"])
    return {"agent1response":response}

# Agent 2 object
agent2obj=Agent2()

# Agent 2 Retrieval node
def agent2_retrivel_node(state:State):
    indices,chunks=agent2obj.retrivel(
        response=state["agent1response"])

    return {"indices":indices,"chunks":chunks}

# Agent 2 LLM node
def agent2_llm_node(state:State):
    response=agent2obj.llm(
        state["indices"],state["chunks"],state["agent1response"])
    return {"agent2response":response}

# Agent 3 node
def agent3_node(state:State):
    response=priority(
        state["agent2response"])
    return {"agent3response":response}

# Agent 4 node
def agent4_node(state:State):
    response=validate(
        state["agent1response"],
        state["agent3response"])

    return {"agent4response":response}


# Agent 5 node
def agent5_node(state:State):
    report={
        "agent3":state["agent3response"],
        "validation":state["agent4response"]}

    response=agent5(report)

    return {"agent5response":response}


workflow=StateGraph(State)

# Nodes
workflow.add_node("agent1",agent1_node)
workflow.add_node("retrivel",agent2_retrivel_node)
workflow.add_node("agent2_llm",agent2_llm_node)
workflow.add_node("agent3",agent3_node)
workflow.add_node("agent4",agent4_node)
workflow.add_node("agent5",agent5_node)


# Edges
workflow.add_edge(START,"agent1")
workflow.add_edge("agent1","retrivel")
workflow.add_edge("retrivel","agent2_llm")
workflow.add_edge("agent2_llm","agent3")
workflow.add_edge("agent3","agent4")
workflow.add_edge("agent4","agent5")
workflow.add_edge("agent5",END)

# Compile
graph=workflow.compile()

# CALL GRAPH

'''result=graph.invoke({
    "report_text":"HbA1c: 8.2% (Reference: 4–5.6%)"})

print(result["agent5response"])'''