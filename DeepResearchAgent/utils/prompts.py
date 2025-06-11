# Pydantic
from typing import Optional, List
from pydantic import BaseModel, Field

## agent prompts

## prompt to grab wikipedia search terms (could be searches on any repository)
search_terms_prompt = """
# Role
You are an expert researcher responsible for deciding which Wikipedia search term best fits the question asked by the user.

# Instructions
Carefully analyze the following question asked by the user and translate it into 3-5 wikipedia search terms consisting of no more than 3 words. Present the output in a structured format.
The search terms should be included in the search_terms list. The reason for selecting each search term should be included in the reasons list.

question: {query}
"""

class SearchTerms(BaseModel):
    """Class to generate search terms"""
    search_terms: List[str] = Field(description="The search terms.")
    reasons: List[str] = Field(description="The reason for selecting these search terms.")


## prompt to summarize the text on a wikipedia page
page_summarizer_prompt = """
# Role
You are an expert report writer responsible for summarizing provided content based on a question asked by a user.

# Instructions
Carefully analyze the provided page content and create a 3-5 paragraph summary focused specifically on the question asked by the user. Present the output in a structured format.
The summary should be included in summary field. Please include the page name used to generate the page in the search_term field. If the provided page content is valuable and worth adding to research for a larger report, please add a value of true in the include field.
Please score this page on a scale from 0-100 based on whether the page content can directly address the provided question.
If the provided page content cannot help answer the provided question, please give a score of 0 in the score field. 

question: {query}
search term: {page}

page content: {page_content}
"""

class PageSummary(BaseModel):
    """Class to summarize existing research"""
    page: str = Field(description="Page name of the wikipedia page summary.")
    summary: str = Field(description="Summary of the wikipedia page.")
    score: int = Field(description="Relevance and value score of the wikipedia page for the final report.")



## prompt to generate a report based on the collected summaries
report_generator_prompt = """
# Role
You are an expert report writer responsible for generating a report using provided contentbased on a question asked by a user.

# Instructions
Carefully analyze the provided content and create a 3-5 paragraph summary focused specifically on the question asked by the user. Present the output in a structured format. The generated report should be included in the report field.
Please provide a report consisting of AT LEAST 3 paragraphs. Provide paragraph headers and use this example between the dashes below for formatting:

-
<b>Header 1</b>
<br><br>
Paragraph 1 text
<br><br><br>
<b>Header 2</b>
<br><br>
Paragraph 2 text
<br><br><br>
<b>Header 3</b>
<br><br>
Paragraph 3 text
<br><br><br>
<b>Header 4</b>
<br><br>
Paragraph 4 text
<br><br><br>
<b>Header 5</b>
<br><br>
Paragraph 5 text
<br><br><br>
-

Please format the report as HTML, but only tags that can appear INSIDE of <body> tags, so no <html>, <body> or other tags that might exist outside of <body> tags.
Do not put dashes or any other puncuation around the Headers in the answer.

question: {query}

content: {summaries}
"""

class GeneratedReport(BaseModel):
    """Class to summarize existing research"""
    report: str = Field(description="The generated report.")


## prompt to summarize the text on a wikipedia page
image_evaluate_prompt = """
# Role
You are an expert journalist capable of finding the most valuable and relevant images for any report or news story.

# Instructions
Carefully analyze the provided list of images and select the TOP 3 most relevant and valuable images based on the provided question. Present the output in a structured format.
The image source provided for each image should be included in the image_src field.
The image caption provided for each image should be included in the image caption field.
If an image is unrelated to the provided question, do not include the image in the list.
Try to return 3 mages, but if no images are related to the provided question, please return empty image_src and caption lists.

question: {query}
image: {image_list}

"""

class ImageEvaluate(BaseModel):
    """Class to evaluate relevance of images"""
    src: List[str] = Field(description="List of relevant image sources.")
    caption: List[str] = Field(description="List of relevant image captions.")


## prompt to verify the report
report_verification_prompt = """
# Role
You are a meticulous professor specializing in reviewing and analyzing reports to evaluate relevance and accuracy.

# Instructions
You have been handed a report. Your job is to verify the report successfully answers the provided question and is internally consistent, clearly sourced, and makes no unsupported claims.
Return false if the report contains verifiable falsehoods.
If the report contains unsupported claims or unsupported material, mark it true, but point out all issues or uncertainties.

question: {query}
report: {report}

"""

class ReportVerify(BaseModel):
    """Class to summarize existing research"""
    verified: bool = Field(description="Whether the report passes the verification test.")
    issues: str = Field(description="Describe the main issues or concerns whether varified or not")
