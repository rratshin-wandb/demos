from openai import OpenAI
import weave
from weave import Model, Dataset
import wandb
import numpy as np
import json
import asyncio
import requests
import litellm
import os, io
import anthropic
# import chromadb
# import jsonpickle
from weave.scorers import HallucinationFreeScorer, SummarizationScorer, ContextEntityRecallScorer, ContextRelevancyScorer
from datetime import date
from pydantic import BaseModel, Field, PrivateAttr, validate_call
from dotenv import load_dotenv
import random
from PIL import Image
from typing import Optional, List, Any, Union
from typing_extensions import TypedDict
from weave.trace.api import get_current_call
from langgraph.types import Send
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver, BaseCheckpointSaver
from langchain.callbacks.manager import CallbackManager
# from langchain.globals import clear_cache
from langchain_openai import ChatOpenAI
import uuid, nest_asyncio
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from .utils.prompts import (
    search_terms_prompt, SearchTerms,
    page_summarizer_prompt, PageSummary,
    report_generator_prompt, GeneratedReport,
    image_evaluate_prompt, ImageEvaluate,
    report_verification_prompt, ReportVerify
)
from .utils.defs import (
    editDbP,
    extract_images_and_captions
)

import wikipedia
from collections import Counter
from PIL import Image
load_dotenv()

# possible models
openai_models = ["gpt-4o-mini", "gpt-4o", "ft:gpt-4o-mini-2024-07-18:weights-biases::BUfazcLq"]



import operator
from typing_extensions import Annotated
# Define our state
class GraphState(TypedDict):
    query: str
    searchresults: int
    # query: Annotated[str, operator.add]
    search_terms: SearchTerms
    pages: list
    images: list
    relevant_summary: Annotated[list, operator.add]
    summary: Annotated[list, operator.add]
    report: GeneratedReport
    verification: ReportVerify


class GenerateSearchTerms:
    def __init__(self, search_terms_client, search_terms_prompt) -> None:
        self.search_terms_client = search_terms_client
        self.search_terms_prompt = search_terms_prompt

    @weave.op(name="GenerateSearchTermsCall")
    def __call__(self, state: GraphState):
        query = state["query"]

        weave_search_terms_prompt = weave.ref("weave:///wandb-pmm/ResearchAgent/object/search_terms_prompt:v1").get()
        search_terms = self.search_terms_client.invoke(weave_search_terms_prompt.format(query=query))

        state["search_terms"] = search_terms
        return state
    
class GetPages:
    @weave.op(name="GetPagesCall")
    def __call__(self, state: GraphState):

        search_terms = state["search_terms"].search_terms
        query = state["query"]
        i_search_results = state["searchresults"]

        a_pages = []
        for s_search_term in search_terms:
            wp = wikipedia.search(query=s_search_term, results=i_search_results, suggestion=True)
            a_pages.append(wp[0])

        counter = Counter()
        for sublist in a_pages:
            unique_items = set(sublist)
            counter.update(unique_items)

        pages = sorted(
            [item for item, count in counter.items()],
            key=lambda x: counter[x],
            reverse=True
        )

        pages = pages[:i_search_results]

        # print("pages [" + str(pages) + "]")
        state["pages"] = pages
        return state
    
class SummarizePage:
    def __init__(self, page_summarizer_client, page_summarizer_prompt) -> None:
        self.page_summarizer_client = page_summarizer_client
        self.page_summarizer_prompt = page_summarizer_prompt

    @weave.op(name="SummarizePageCall")
    def __call__(self, state: GraphState):
        page = state["page"]
        query = state["query"]
        wpp = None
        try:
            wpp = wikipedia.page(page, auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            random_page = random.choice(e.options)
            wpp = wikipedia.page(random_page)
        page_content = remove_after_substring(str(wpp.content), "== See also ==");
        summary = self.page_summarizer_client.invoke(self.page_summarizer_prompt.format(page=page, query=query, page_content=page_content))
        state["summary"] = summary
        result = { "summary" : [summary.model_dump()] }
        if (summary.score >= 75):
            result["relevant_summary"] = [summary.model_dump()]
        return result
    
class EvaluateImages:
    def __init__(self, image_evaluate_client, image_evaluate_prompt) -> None:
        self.image_evaluate_client = image_evaluate_client
        self.image_evaluate_prompt = image_evaluate_prompt

    @weave.op(name="EvaluateImagesCall")
    def __call__(self, state: GraphState):
        query = state["query"]
        summary = state["relevant_summary"]
        sorted_relevant_summary = sorted(summary, key=lambda x: x['score'], reverse=True)
        image_list = ""
        page = ""
        for s in sorted_relevant_summary:
            page = wikipedia.page(s["page"], auto_suggest=False)
            s_html = page.html()
            a_images = extract_images_and_captions(s_html)
            for o_image in a_images:
                image_list += "image src: " + str(o_image["src"]) + "\n" + "image caption: " + str(o_image["caption"]) + "\n\n"

        images = self.image_evaluate_client.invoke(self.image_evaluate_prompt.format(page=page, query=query, image_list=image_list))

        return {
            "images" : [images.model_dump()]
        }
    
class GenerateReport:
    def __init__(self, report_generator_client, report_generator_prompt) -> None:
        self.report_generator_client = report_generator_client
        self.report_generator_prompt = report_generator_prompt

    @weave.op(name="GenerateReportCall")
    def __call__(self, state: GraphState):

        query = state["query"]
        summaries = '\n\n'.join(map(str, state["summary"]))
        weave_report_generator_prompt = weave.ref("weave:///wandb-pmm/ResearchAgent/object/report_generator_prompt:Mq5fd997Ofq4EfEHeqHb882tDH5bUAo8Ng1hlQJFHg0").get()
        report = self.report_generator_client.invoke(weave_report_generator_prompt.format(query=query, summaries=summaries))
        a_image_src = state["images"][0]["src"]
        a_image_caption = state["images"][0]["caption"]
        if len(a_image_src) > 0:
            s_html_images_div = "<div style=\"float: left; width: 100%;\">"
            s_html_image_div = ""
            for i in range(len(a_image_src)):
                s_html_image_div += "<div style=\"float: left; width: calc(33% - 10px);\">"
                s_html_image_div += "<div style=\"float: left; width: 100%;\"><a href=\"" + a_image_src[i] + "\" target=\"_blank\" style=\"text-decoration: none;\"><img src=\"" + a_image_src[i] + "\" style=\"border: solid 1px #999999; width: 100%;\"></a></div>"
                s_html_image_div += "<div style=\"clear: both; height: 4px; width: 1px;\"></div>"
                s_html_image_div += "<div style=\"float: left; width: 100%; font-size: 11px; line-height: 12px\">"+ a_image_caption[i] + "</div>"
                s_html_image_div += "</div>"
                if i < (len(a_image_src) - 1):
                    s_html_image_div += "<div style=\"float: left; width: 15px; height: 1px;\"></div>"
            s_html_image_div += "<div style=\"clear: both; height: 30px; width: 1px;\"></div>"
            s_html_images_div += s_html_image_div + "</div>"
            report.report = s_html_images_div + report.report
        state["report"] = report
        return {
            "report" : report
        }

class VerifyReport:
    def __init__(self, report_verification_client, report_verification_prompt) -> None:
        self.report_verification_client = report_verification_client
        self.report_verification_prompt = report_verification_prompt

    @weave.op(name="VerifyReportCall")
    def __call__(self, state: GraphState):

        report = state["report"].report
        query = state["query"]
        verification = self.report_verification_client.invoke(self.report_verification_prompt.format(report=report, query=query))
        return {
            "verification" : verification
        }
    
@weave.op
def send_search_terms(state: GraphState):
    return [Send("summarize_page", {"query" : state["query"], "search_term": s}) for s in state["search_terms"].search_terms]

@weave.op
def send_pages(state: GraphState):
    return [Send("summarize_page", {"query" : state["query"], "page": p}) for p in state["pages"]]

@weave.op
def create_wf(s_mode, s_model):

    i_summarizers = 3;

    # Create our graph
    workflow = StateGraph(GraphState)

    # Add nodes to our graph
    workflow.add_node("generate_search_terms", GenerateSearchTerms(search_terms_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(SearchTerms), search_terms_prompt=search_terms_prompt))
    workflow.add_node("get_pages", GetPages())
    workflow.add_node("summarize_page", SummarizePage(page_summarizer_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(PageSummary), page_summarizer_prompt=page_summarizer_prompt))
    workflow.add_node("evaluate_images", EvaluateImages(image_evaluate_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(ImageEvaluate), image_evaluate_prompt=image_evaluate_prompt))
    workflow.add_node("generate_report", GenerateReport(report_generator_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(GeneratedReport), report_generator_prompt=report_generator_prompt))
    if s_mode == "verify":
        workflow.add_node("verify_report", VerifyReport(report_verification_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(ReportVerify), report_verification_prompt=report_verification_prompt))

    # # Connect our nodes
    workflow.set_entry_point("generate_search_terms")
    workflow.add_edge("generate_search_terms", "get_pages")
    workflow.add_conditional_edges("get_pages", send_pages, ["summarize_page"])
    workflow.add_edge("summarize_page", "evaluate_images")
    workflow.add_edge("evaluate_images", "generate_report")
    if s_mode == "verify":
        workflow.add_edge("generate_report", "verify_report")
        workflow.add_edge("verify_report", END)
    else:
        workflow.add_edge("generate_report", END)

    # Compile our graph
    app = workflow.compile()
    return app

def create_wf_display(i_count_pages, s_mode):
    s_model = "gpt-4.1-mini"
    workflow = StateGraph(GraphState)
    workflow.add_node("generate_search_terms", GenerateSearchTerms(search_terms_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(SearchTerms), search_terms_prompt=search_terms_prompt))
    workflow.add_node("get_pages", GetPages())
    b_greater_than_4 = False
    i_count_pages_display = i_count_pages
    if i_count_pages > 4:
        b_greater_than_4 = True
        i_count_pages_display = 5
    for i in range(i_count_pages_display):
        i_index = (i + 1)
        if b_greater_than_4 and i_index == 5:
            i_index = "n..." + str(i_count_pages)
        workflow.add_node("summarize_page_" + str(i_index), SummarizePage(page_summarizer_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(PageSummary), page_summarizer_prompt=page_summarizer_prompt))
    workflow.add_node("evaluate_images", EvaluateImages(image_evaluate_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(ImageEvaluate), image_evaluate_prompt=image_evaluate_prompt))
    workflow.add_node("generate_report", GenerateReport(report_generator_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(GeneratedReport), report_generator_prompt=report_generator_prompt))
    if s_mode == "verify":
        workflow.add_node("verify_report", VerifyReport(report_verification_client=ChatOpenAI(model=s_model, max_retries=5).with_structured_output(ReportVerify), report_verification_prompt=report_verification_prompt))


    # # Connect our nodes
    workflow.set_entry_point("generate_search_terms")
    workflow.add_edge("generate_search_terms", "get_pages")
    for i in range(i_count_pages_display):
        i_index = (i + 1)
        if b_greater_than_4 and i_index == 5:
            i_index = "n..." + str(i_count_pages)
        workflow.add_edge("get_pages", "summarize_page_" + str(i_index))
        workflow.add_edge("summarize_page_" + str(i_index), "evaluate_images")
    workflow.add_edge("evaluate_images", "generate_report")
    if s_mode == "verify":
        workflow.add_edge("generate_report", "verify_report")
        workflow.add_edge("verify_report", END)
    else:
        workflow.add_edge("generate_report", END)
    app = workflow.compile()
    return app

@weave.op
def draw_graph(i_count_pages, s_mode):
    app = create_wf_display(i_count_pages, s_mode)
    # Assuming you have already created and compiled your graph as 'app'
    png_graph = app.get_graph().draw_mermaid_png()
    img = Image.open(io.BytesIO(png_graph))
    return img

@weave.op
# def graph_stream(app, i_analysis_id):
def graph_stream(app, i_analysis_id, s_mode, query, i_search_results):
    result = {}
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Start a new event loop
    print("Starting deep research workflow...")
    s_query = "update t_research_analyses set progress = progress || '\n' || %s where analysisid = " + str(i_analysis_id)
    a_values = (["Starting deep research workflow..."])
    editDbP(s_query, a_values)
    result["query"] = query

    try:
        # Execute the prediction
        inputs = {"query": query, "searchresults" : i_search_results}
        for event in app.stream(inputs, config=thread_config):
            for key, value in event.items():
                s_query = "update t_research_analyses set progress = progress || '\n' || %s where analysisid = " + str(i_analysis_id)
                a_values = ([key])
                editDbP(s_query, a_values)
                if key == "generate_search_terms":
                    result["search_terms"] = value.get("search_terms", "")
                if key == "get_pages":
                    result["pages"] = value.get("pages", "")
                if key == "evaluate_images":
                    result["images"] = value.get("images", "")
                if key == "generate_report":
                    result["report"] = value.get("report", "").report
                if s_mode == "verify" and key == "verify_report":
                    result["verification"] = value.get("verification", "")

        
    except Exception as e:
        print(f"Error in prediction workflow: {str(e)}")

    return result

def remove_after_substring(text, substring):
    index = text.find(substring)
    if index != -1:
        return text[:index]
    return text

class ResearchAgent(weave.Model):

    @weave.op
    def predict(self, i_analysis_id: int, s_mode:str, s_query:str, user:str, s_model:str, i_search_results:int) -> dict:

        # run graph
        app = create_wf(s_mode, s_model)

        # result = graph_stream(app, i_analysis_id)
        result = graph_stream(app, i_analysis_id, s_mode, s_query, i_search_results)
        result["diagram"] = draw_graph(len(result["pages"]), s_mode)

        if s_mode == "verify":
            b_verified = result["verification"].verified
            s_verification_issues = result["verification"].issues
            result["report"] += "<br><br><hr><br><i>Verified: " + str(b_verified) + "<br><br>" + s_verification_issues + "</i><br><br><hr><br>"

        s_query = "update t_research_analyses set status = 'complete', report = %s where analysisid = " + str(i_analysis_id)
        a_values = ([result["report"]])
        editDbP(s_query, a_values)


        return result







