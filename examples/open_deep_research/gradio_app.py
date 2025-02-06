#!/usr/bin/env python

import os
from dotenv import load_dotenv

import gradio as gr

import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("gradio").setLevel(logging.DEBUG)

# Load environment variables (e.g. HF_TOKEN, SERPAPI_API_KEY, etc.)
load_dotenv()

from smolagents import (
    MANAGED_AGENT_PROMPT,
    CodeAgent,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
    GradioUI,
)

from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

def create_agent_hierarchy(model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        managed_agent_prompt=MANAGED_AGENT_PROMPT
        + """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.""",
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent

def main():
    # Create a model instance.
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
        max_completion_tokens=8192,
    )

    # Build the agent hierarchy (which sets up the manager agent and its sub-agents)
    agent = create_agent_hierarchy(model)

    # Option 1: Use the built-in GradioUI from smolagents for an interactive multi‐step interface.
    ui = GradioUI(agent)
    ui.launch()

    # Option 2: Use a simpler Q&A interface that wraps a function that sends a question to the agent.
    # (Uncomment the code below to use this alternative.)
    # def ask_agent(question: str) -> str:
    #     # Run the agent with the provided question.
    #     # (This call is blocking until the agent produces a final answer.)
    #     answer = agent.run(question)
    #     return str(answer)
    #
    # iface = gr.Interface(
    #     fn=ask_agent,
    #     inputs=gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
    #     outputs="text",
    #     title="Agent Q&A",
    #     description="Ask a question to kickstart the agent and get an answer.",
    # )
    # iface.launch()

if __name__ == "__main__":
    main()
