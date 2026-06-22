---
title: Deep Research
description: "Autonomous deep research with AG2 using DeepResearchTool and GPT Researcher. Multi-step research tasks with Tavily-powered web search and structured report generation."
---

AG2 recommends two approaches for autonomous deep research: the built-in `DeepResearchTool` and the [GPT Researcher](https://docs.gptr.dev/) multi-agent integration.

**DeepResearchTool** is a composable AG2 tool that any agent can call directly. It is the right choice when you want research capabilities tightly integrated into your AG2-based agent, without a separate orchestration layer.

**GPT Researcher** is a standalone research agent with a built-in orchestration pipeline optimized for recursive querying and structured report generation. It is a good choice when you want a pre-built, highly configurable research agent that can operate independently or alongside other AG2 agents.

## DeepResearchTool

Inspired by [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research), `DeepResearchTool` is designed to tackle complex, multi-step research tasks, synthesizing insights from diverse online sources.

### Installation

<Warning>
`DeepResearchTool` is built on top of [`Browser Use`](https://github.com/browser-use/browser-use), which requires **Python 3.11 or higher**.
</Warning>

To get started with the `DeepResearchTool`, follow these steps:

1. Install AG2 with the `browser-use` extra:
    ```bash
    pip install ag2[openai,browser-use]
    ```
    <Tip>
    If you have been using `autogen` or `ag2`, all you need to do is upgrade it using:
    ```bash
    pip install -U autogen[openai,browser-use]
    ```
    or
    ```bash
    pip install -U ag2[openai,browser-use]
    ```
    as `autogen` and `ag2` are aliases for the same PyPI package.
    </Tip>
2. Set up Playwright:

    ```bash
    # Installs Playwright and browsers for all OS
    playwright install
    # Additional command, mandatory for Linux only
    playwright install-deps
    ```

3. For running the code in Jupyter, use `nest_asyncio` to allow nested event loops.
    ```bash
    pip install nest_asyncio
    ```

You're all set! Now you can start using browsing features in AG2.

### Imports

```python
import os
import nest_asyncio

from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from autogen.tools.experimental import DeepResearchTool

nest_asyncio.apply()
```

### Agent Configuration

Configure the agents for the interaction.

- `config_list` defines the LLM configurations, including the model and API key.
- `UserProxyAgent` simulates user inputs without requiring actual human interaction (set to `NEVER`).
- `AssistantAgent` represents the AI agent, configured with the LLM settings.

```python
llm_config = LLMConfig(config_list={
    "api_type": "openai",
    "model": "gpt-5",
    "api_key": os.environ["OPENAI_API_KEY"],
})

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

assistant = AssistantAgent(name="assistant", llm_config=llm_config)
```

### DeepResearchTool Configuration

Once the `DeepResearchTool` is created, it needs to be registered to the agents.

```python
deep_research_tool = DeepResearchTool(llm_config=llm_config)
deep_research_tool.register_for_execution(user_proxy)
deep_research_tool.register_for_llm(assistant)
```

### Initiate Chat

```python
result = user_proxy.initiate_chat(
    recipient=assistant,
    message="What was the impact of DeepSeek on stock prices and why?",
    max_turns=2,
)
```

```console
user_proxy (to assistant):

What was the impact of DeepSeek on stock prices and why?

--------------------------------------------------------------------------------
assistant (to user_proxy):

***** Suggested tool call (call_JqTQWaKKfCipuIqXTBSMl6jM): delegate_research_task *****
Arguments:
{"task":"Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects."}
***************************************************************************************

--------------------------------------------------------------------------------

>>>>>>>> EXECUTING FUNCTION delegate_research_task...
Call ID: call_JqTQWaKKfCipuIqXTBSMl6jM
Input arguments: {'task': 'Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects.'}
CriticAgent (to SummarizerAgent):

Please answer the following question: Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects.

--------------------------------------------------------------------------------
SummarizerAgent (to CriticAgent):

***** Suggested tool call (call_Wqo4b8ExbFNxbQcw4Vyh9Xwl): split_question_and_answer_subquestions *****
Arguments:
{"question":"Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects."}
*******************************************************************************************************

--------------------------------------------------------------------------------

>>>>>>>> EXECUTING FUNCTION split_question_and_answer_subquestions...
Call ID: call_Wqo4b8ExbFNxbQcw4Vyh9Xwl
Input arguments: {'question': 'Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects.'}
DecompositionCritic (to DecompositionAgent):

Analyse and gather subqestions for the following question: Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects.

--------------------------------------------------------------------------------
DecompositionAgent (to DecompositionCritic):

1. What is DeepSeek search technology and how does it function in the context of financial markets?
2. How has the introduction of DeepSeek search technology coincided with changes in stock prices, both immediately and over the longer term?
3. Which specific stock prices or market indices have been most affected by DeepSeek search technology, and how do their changes compare?
4. What are the potential mechanisms or factors by which DeepSeek search technology could influence stock prices?
5. How has investor and market behavior reacted to the deployment of DeepSeek search technology, and what reasoning supports these reactions?

--------------------------------------------------------------------------------
DecompositionCritic (to DecompositionAgent):

The subquestions provided cover distinct aspects of the investigation into the impact of DeepSeek search technology on stock prices. However, some overlap and redundancies can be identified. Here’s a refined list:

1. What is DeepSeek search technology and how does it function in the context of financial markets?
2. How has the introduction of DeepSeek search technology coincided with changes in stock prices?
3. Which specific stocks or sectors have demonstrated significant changes attributed to DeepSeek search technology?
4. What mechanisms or factors are responsible for DeepSeek search technology's influence on stock prices?

These subquestions cover the identification, impact, and mechanisms of the observed stock price changes without unnecessary repetition. Now, let's prepare the subquestions for processing.

I will now proceed with generating subquestions.
***** Suggested tool call (call_qOxMZ9wptmjcrn0Tu2VO3KQn): generate_subquestions *****
Arguments:
{"task":{"question":"Investigate the impact of DeepSeek search technology on stock prices and provide detailed reasoning for the observed effects.","subquestions":[{"question":"What is DeepSeek search technology and how does it function in the context of financial markets?"},{"question":"How has the introduction of DeepSeek search technology coincided with changes in stock prices?"},{"question":"Which specific stocks or sectors have demonstrated significant changes attributed to DeepSeek search technology?"},{"question":"What mechanisms or factors are responsible for DeepSeek search technology's influence on stock prices?"}]}}
**************************************************************************************

...

assistant (to user_proxy):

DeepSeek search technology significantly impacted stock prices, particularly in the technology sector, by introducing increased market volatility and substantial fluctuations. Here's a detailed breakdown:

### Overview
- **Impact of DeepSeek**: This AI-driven search technology from China caused disruptions in financial markets, particularly affecting U.S. tech stocks. Its introduction led to massive market capitalization losses, as seen in a trillion-dollar reduction in the Nasdaq index, heavily composed of tech companies.

### How DeepSeek Functions
- **Cost-Effectiveness and Innovation**: DeepSeek's V3 version introduced a highly cost-effective AI model that challenged the existing dominance of international tech firms. This led investors to reevaluate their holdings in these companies, fearing DeepSeek's potential to capture significant market share.

### Changes in Stock Prices
- **Tech Sector Impact**: The technology sector witnessed massive sell-offs, with companies like Nvidia facing severe drops in stock value. The overall tech industry lost billions of dollars following the DeepSeek announcements.
- **Software Stocks**: Some software stocks, perceived as beneficiaries of this shift, experienced positive investor sentiment contrary to other declining AI stocks.

### Mechanisms Influencing Stock Prices
1. **Market Erosion**: DeepSeek’s operational cost advantages and open-source nature disrupted major AI competitors, eroding their market share and prompting a reevaluation of tech stock valuations.
2. **Valuation Reassessments**: Concerns over tech stock valuations heightened market anxiety, driven by competitive threats from DeepSeek.

### Conclusion
Overall, DeepSeek challenged tech giants globally, leading to valuation adjustments and increased stock market volatility. This necessitated a broader reassessment of the technological landscape by investors, culminating in significant financial impacts.

This comprehensive understanding explains the observed effects on stock prices due to DeepSeek. TERMINATE

--------------------------------------------------------------------------------
```

<DeepResearch/>

## GPT Researcher

[GPT Researcher](https://docs.gptr.dev/) is an autonomous research agent that orchestrates a team of specialized agents to produce comprehensive research reports. The [AG2 integration](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/ag2) uses AG2 as the orchestration layer for the research pipeline.

### Research Team

The pipeline coordinates eight specialized agents:

- **Chief Editor** — manages the team and oversees the research process
- **Researcher** — conducts autonomous, in-depth topic investigation using GPT Researcher
- **Editor** — plans the research structure and outline
- **Reviewer** — validates results against established criteria
- **Revisor** — refines findings based on reviewer feedback
- **Writer** — compiles the final report
- **Publisher** — outputs the report in multiple formats (Markdown, PDF, Docx)
- **Human** — provides optional oversight and feedback at any stage

### Installation

Clone the GPT Researcher repository and install the dependencies:

```bash
git clone https://github.com/assafelovic/gpt-researcher.git
cd gpt-researcher
pip install -r requirements.txt
pip install -r multi_agents_ag2/requirements.txt
```

### Environment Setup

```bash
export OPENAI_API_KEY={your_openai_key}
export TAVILY_API_KEY={your_tavily_key}
```

A [Tavily API key](https://tavily.com/) is required for web search. See [Tavily Search Tool](tavily-search.md) for setup details and standalone usage with AG2 agents.

### Configure the Research Task

Customize the research task by editing `multi_agents_ag2/task.json`:

```json
{
  "query": "Is AI in a hype cycle?",
  "max_sections": 3,
  "max_revisions": 3,
  "publish_formats": {
    "markdown": true,
    "pdf": true,
    "docx": true
  },
  "include_human_feedback": false,
  "follow_guidelines": false,
  "model": "gpt-4o",
  "guidelines": [
    "The report MUST be written in APA format",
    "Each sub section MUST include supporting sources using hyperlinks."
  ],
  "verbose": true
}
```

| Parameter | Description |
|---|---|
| `query` | The research question |
| `max_sections` | Maximum number of report sections |
| `max_revisions` | Maximum revision cycles per section |
| `publish_formats` | Output formats: `markdown`, `pdf`, `docx` |
| `include_human_feedback` | Whether to prompt for human input during research |
| `model` | LLM model to use |
| `follow_guidelines` | Whether to enforce the `guidelines` list |

### Run the Research

```bash
python -m multi_agents_ag2.main
```

The pipeline runs through five stages: planning, data gathering and analysis, quality review and revision, writing, and publishing.

### Resources

- [GPT Researcher documentation](https://docs.gptr.dev/)
- [AG2 integration guide](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/ag2)
- [Source code on GitHub](https://github.com/assafelovic/gpt-researcher/tree/main/multi_agents_ag2)
