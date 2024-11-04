# src/components.py
import json
import operator
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from llm_models import llm
from vectorstore import VectorStoreManager
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from IPython.display import Image, display

# Prompt and instructions setup
router_instructions = """You are an expert at Swedish and winter road maintenance employed by Klimator.

The vectorstore contains documents related to meteorological facts, winter road maintenance laws, Road Weather Forecast, and related fields of study.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with a single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

doc_grader_instructions = """You are a grader assessing the relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with a single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

rag_prompt = """You are an assistant for question-answering tasks in Swedish. 

Here is the context to use to answer the question:

{context}

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this question using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

hallucination_grader_instructions = """You are a teacher grading a quiz in Swedish. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

answer_grader_instructions = """You are a teacher grading a quiz in Swedish. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """Graph state containing information for each graph node."""
    question: str
    generation: str
    web_search: str
    max_retries: int
    answers: int
    loop_step: int
    documents: List[str]

# Functions for workflow
def initialize_graph(config):
    """Define graph, nodes, and edges as per the logic."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("websearch", web_search)  # web search
    graph.add_node("retrieve", retrieve)  # retrieve
    graph.add_node("grade_documents", grade_documents)  # grade documents
    graph.add_node("generate", generate)  # generate

    # Build graph
    graph.set_conditional_entry_point(route_question, {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    })
    graph.add_edge("websearch", "generate")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", decide_to_generate, {
        "websearch": "websearch",
        "generate": "generate",
    })
    graph.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    })

    # Compile the graph
    return graph.compile()

# Implementations of nodes
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("---RETRIEVE---")
    question = state["question"]        
    documents = VectorStoreManager.retrieve_documents(question)  # Assuming 'retriever' is defined elsewhere
    return {"documents": documents}

def generate(state):
    """Generate answer using RAG on retrieved documents."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state):
    """Grade the relevance of retrieved documents to the question."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    relevant_doc_found = False

    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_doc_found = True
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    web_search = "No" if relevant_doc_found else "Yes"
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    """Web search based on the question."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}

def route_question(state):
    """Route question to web search or RAG."""
    print("---ROUTE QUESTION---")
    route_question = llm.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    return "websearch" if source == "websearch" else "retrieve"

def decide_to_generate(state):
    """Determine whether to generate an answer or run web search."""
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        print("---DECISION: INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """Determine whether the generation is grounded in the document and answers question."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    hall_score = json.loads(result.content)
    state["hallucination_score"] = hall_score["binary_score"]
    state["hallucination_explanation"] = hall_score["explanation"]
    return {
        "score": hall_score["binary_score"],
        "explanation": hall_score["explanation"],
    }

def grade_answer(state):
    """Grade student answer against the question."""
    print("---GRADE STUDENT ANSWER---")
    question = state["question"]
    generation = state["generation"]
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        question=question, generation=generation.content
    )
    result = llm.invoke(
        [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
    )
    score = json.loads(result.content)
    return score

def run_workflow(state: Dict[str, Any], config: Dict[str, Any]):
    """Run the main workflow."""
    graph = initialize_graph(config)
    return graph(state)