<a name="readme-top"></a>

![Pypi Downloads](https://img.shields.io/pypi/dm/pyautogen?label=PyPI%20downloads)
[![PyPI version](https://badge.fury.io/py/autogen.svg)](https://badge.fury.io/py/autogen)
[![Build](https://github.com/ag2ai/ag2/actions/workflows/python-package.yml/badge.svg)](https://github.com/ag2ai/ag2/actions/workflows/python-package.yml)
![Python Version](https://img.shields.io/badge/3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![Discord](https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat)](https://discord.gg/pAbnFJrkgZ)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ag2ai)](https://x.com/ag2oss)

<!-- [![NuGet version](https://badge.fury.io/nu/AutoGen.Core.svg)](https://badge.fury.io/nu/AutoGen.Core) -->

# [AG2](https://github.com/ag2ai/ag2)

## Key Features
- 🤖 Multi-Agent Framework - Build and orchestrate AI agent interactions
- 🔧 Flexible Integration - Support for various LLMs (OpenAI, Anthropic, Gemini, etc.)
- 🛠 Tool Usage - Agents can use external tools and execute code
- 👥 Human-in-the-Loop - Seamless human participation when needed
- 🔄 Rich Orchestration Patterns - Agents can be organized in any form you like
- 🎯 Future-Oriented - Designed for solving difficult problems and harnessing latest and future technology

[📚 Documentation](https://docs.ag2.ai/) | [💡 Examples](https://github.com/ag2ai/build-with-ag2) | [🤝 Contributing](https://docs.ag2.ai/docs/contributor-guide/contributing)

[📚 Cite paper](#related-papers).
<!-- <p align="center">
    <img src="https://github.com/ag2ai/ag2/blob/main/website/static/img/flaml.svg"  width=200>
    <br>
</p> -->

> **:tada: IMPORTANT**
>
> :fire: :tada: **Nov 11, 2024:** We are evolving AutoGen into **AG2**!
> A new organization [AG2AI](https://github.com/ag2ai) is created to host the development of AG2 and related projects with open governance. Check [AG2's new look](https://ag2.ai/).
>
> We invite collaborators from all organizations and individuals to join the development.


:fire: :tada: AG2 is available via `pyautogen` (or its alias  `autogen` or  `ag2`)  on PyPI!

```
pip install pyautogen
```

📄 **License:**
We adopt the Apache 2.0 license from v0.3. This enhances our commitment to open-source collaboration while providing additional protections for contributors and users alike.


:tada: May 29, 2024: DeepLearning.ai launched a new short course [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen), made in collaboration with Microsoft and Penn State University, and taught by AutoGen creators [Chi Wang](https://github.com/sonichi) and [Qingyun Wu](https://github.com/qingyun-wu).

:tada: May 24, 2024: Foundation Capital published an article on [Forbes: The Promise of Multi-Agent AI](https://www.forbes.com/sites/joannechen/2024/05/24/the-promise-of-multi-agent-ai/?sh=2c1e4f454d97) and a video [AI in the Real World Episode 2: Exploring Multi-Agent AI and AutoGen with Chi Wang](https://www.youtube.com/watch?v=RLwyXRVvlNk).

:tada: May 13, 2024: [The Economist](https://www.economist.com/science-and-technology/2024/05/13/todays-ai-models-are-impressive-teams-of-them-will-be-formidable) published an article about multi-agent systems (MAS) following a January 2024 interview with [Chi Wang](https://github.com/sonichi).

:tada: May 11, 2024: [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://openreview.net/pdf?id=uAjxFFing2) received the best paper award at the [ICLR 2024 LLM Agents Workshop](https://llmagents.github.io/).

<!-- :tada: Apr 26, 2024: [AutoGen.NET](https://docs.ag2.ai/ag2-for-net/) is available for .NET developers! -->

:tada: Apr 17, 2024: Andrew Ng cited AutoGen in [The Batch newsletter](https://www.deeplearning.ai/the-batch/issue-245/) and [What's next for AI agentic workflows](https://youtu.be/sal78ACtGTc?si=JduUzN_1kDnMq0vF) at Sequoia Capital's AI Ascent (Mar 26).

:tada: Mar 3, 2024: What's new in AutoGen? 📰[Blog](https://docs.ag2.ai/blog/2024-03-03-AutoGen-Update); 📺[Youtube](https://www.youtube.com/watch?v=j_mtwQiaLGU).

<!-- :tada: Mar 1, 2024: the first AutoGen multi-agent experiment on the challenging [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) benchmark achieved the No. 1 accuracy in all the three levels. -->

<!-- :tada: Jan 30, 2024: AutoGen is highlighted by Peter Lee in Microsoft Research Forum [Keynote](https://t.co/nUBSjPDjqD). -->

:tada: Dec 31, 2023: [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://arxiv.org/abs/2308.08155) is selected by [TheSequence: My Five Favorite AI Papers of 2023](https://thesequence.substack.com/p/my-five-favorite-ai-papers-of-2023).

<!-- :fire: Nov 24: pyautogen [v0.2](https://github.com/ag2ai/ag2/releases/tag/v0.2.0) is released with many updates and new features compared to v0.1.1. It switches to using openai-python v1. Please read the [migration guide](https://docs.ag2.ai/docs/installation/Installation). -->

<!-- :fire: Nov 11: OpenAI's Assistants are available in AutoGen and interoperatable with other AutoGen agents! Checkout our [blogpost](https://docs.ag2.ai/blog/2023-11-13-OAI-assistants) for details and examples. -->

:tada: Nov 8, 2023: AutoGen is selected into [Open100: Top 100 Open Source achievements](https://www.benchcouncil.org/evaluation/opencs/annual.html) 35 days after spinoff from [FLAML](https://github.com/microsoft/FLAML).

<!-- :tada: Nov 6, 2023: AutoGen is mentioned by Satya Nadella in a [fireside chat](https://youtu.be/0pLBvgYtv6U). -->

<!-- :tada: Nov 1, 2023: AutoGen is the top trending repo on GitHub in October 2023. -->

<!-- :tada: Oct 03, 2023: AutoGen spins off from [FLAML](https://github.com/microsoft/FLAML) on GitHub. -->

<!-- :tada: Aug 16: Paper about AutoGen on [arxiv](https://arxiv.org/abs/2308.08155). -->

:tada: Mar 29, 2023: AutoGen is first created in [FLAML](https://github.com/microsoft/FLAML).

<!--
:fire: FLAML is highlighted in OpenAI's [cookbook](https://github.com/openai/openai-cookbook#related-resources-from-around-the-web).

:fire: [autogen](https://docs.ag2.ai/) is released with support for ChatGPT and GPT-4, based on [Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673).

:fire: FLAML supports Code-First AutoML & Tuning – Private Preview in [Microsoft Fabric Data Science](https://learn.microsoft.com/en-us/fabric/data-science/). -->

## What is AG2

AG2 (formerly AutoGen) is an open-source AgentOS for building AI agents and facilitating cooperation among multiple agents to solve tasks. AG2 provides fundamental building blocks needed to create, deploy, and manage AI agents that can work together to solve complex problems.

### Core Concepts
- **Agents**: Stateful entities that can send messages, receive messages, and generate replies using underlying capabilities powered by LLMs, non-LLM tools, or human inputs. Depending on the underlying capability, an agent may reason, plan, execute tasks or involve other agents before generating a reply.
- **Conversations**: Structured communication patterns between agents.

**Open Source Statement**: The project welcomes contributions from developers and organizations worldwide. Our goal is to foster a collaborative and inclusive community where diverse perspectives and expertise can drive innovation and enhance the project's capabilities. Whether you are an individual contributor or represent an organization, we invite you to join us in shaping the future of this project. Together, we can build something truly remarkable.

The project is currently maintained by a [dynamic group of volunteers](MAINTAINERS.md) from several organizations. Contact project administrators Chi Wang and Qingyun Wu via [support@ag2.ai](mailto:support@ag2.ai) if you are interested in becoming a maintainer.


![AG2 Overview](https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/website/static/img/autogen_agentchat.png)


<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

<!--
## Roadmaps
-->

## Quickstart
The easiest way to start playing is
1. Click below to use the GitHub Codespace

    [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ag2ai/ag2?quickstart=1)

 2. Copy OAI_CONFIG_LIST_sample to ./notebook folder, name to OAI_CONFIG_LIST, and set the correct configuration.
 3. Start playing with the notebooks!

*NOTE*: OAI_CONFIG_LIST_sample lists gpt-4o as the default model. If you use a different model, you may need to revise various system prompts (especially if using weaker models like gpt-4o-mini). Proceed with caution when updating this default and be aware of additional risks related to alignment and safety.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## [Installation](https://docs.ag2.ai/docs/installation/Installation)

### Option 1. Install and Run AG2 in Docker

Find detailed instructions for users [here](https://docs.ag2.ai/docs/installation/Docker#step-1-install-docker), and for developers [here](https://docs.ag2.ai/docs/contributor-guide/docker).

### Option 2. Install AG2 Locally

AG2 requires **Python version >= 3.9, < 3.14**. It can be installed from pip:

```bash
pip install ag2
```

Minimal dependencies are installed without extra options. You can install extra options based on the feature you need.

<!-- For example, use the following to install the dependencies needed by the [`blendsearch`](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#blendsearch-economical-hyperparameter-optimization-with-blended-search-strategy) option.
```bash
pip install "autogen[blendsearch]"
``` -->

Find more options in [Installation](https://docs.ag2.ai/docs/Installation#option-2-install-autogen-locally-using-virtual-environment).

<!-- Each of the [`notebook examples`](https://github.com/ag2ai/ag2/tree/main/notebook) may require a specific option to be installed. -->

Even if you are installing and running AG2 locally outside of docker, the recommendation and default behavior of agents is to perform [code execution](https://docs.ag2.ai/docs/FAQ#if-you-want-to-run-code-execution-in-docker) in docker. Find more instructions and how to change the default behaviour [here](https://docs.ag2.ai/docs/FAQ#if-you-want-to-run-code-execution-locally).

For LLM inference configurations, check the [FAQs](https://docs.ag2.ai/docs/FAQ#set-your-api-endpoints).

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Multi-Agent Conversation Framework

AG2 enables the next-gen LLM applications with a generic [multi-agent conversation](https://docs.ag2.ai/docs/Use-Cases/agent_chat) framework. It offers customizable and conversable agents that integrate LLMs, tools, and humans.
By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code.

Features of this use case include:

- **Multi-agent conversations**: AG2 agents can communicate with each other to solve tasks. This allows for more complex and sophisticated applications than would be possible with a single LLM.
- **Customization**: AG2 agents can be customized to meet the specific needs of an application. This includes the ability to choose the LLMs to use, the types of human input to allow, and the tools to employ.
- **Human participation**: AG2 seamlessly allows human participation. This means that humans can provide input and feedback to the agents as needed.

For [example](https://github.com/ag2ai/ag2/blob/main/test/twoagent.py),

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# See https://docs.ag2.ai/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
# You can also set config_list directly as a list, for example, config_list = [{'model': 'gpt-4o', 'api_key': '<your OpenAI API key here>'},]
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False}) # IMPORTANT: set to True to run code in docker, recommended
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

This example can be run with

```python
python test/twoagent.py
```

After the repo is cloned.
The figure below shows an example conversation flow with AG2.

![Agent Chat Example](https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/website/static/img/chat_example.png)


Alternatively, the [sample code](https://github.com/ag2ai/build-with-ag2/blob/main/samples/simple_chat.py) here allows a user to chat with an AG2 agent in ChatGPT style.
Please find more [code examples](https://docs.ag2.ai/docs/Examples#automated-multi-agent-chat) for this feature.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Enhanced LLM Inferences

AG2 also helps maximize the utility out of the expensive LLMs such as gpt-4o. It offers [enhanced LLM inference](https://docs.ag2.ai/docs/Use-Cases/enhanced_inference#api-unification) with powerful functionalities like caching, error handling, multi-config inference and templating.

<!-- For example, you can optimize generations by LLM with your own tuning data, success metrics, and budgets.

```python
# perform tuning for openai<1
config, analysis = autogen.Completion.tune(
    data=tune_data,
    metric="success",
    mode="max",
    eval_func=eval_func,
    inference_budget=0.05,
    optimization_budget=3,
    num_samples=-1,
)
# perform inference for a test instance
response = autogen.Completion.create(context=test_instance, **config)
```

Please find more [code examples](https://docs.ag2.ai/docs/Examples#tune-gpt-models) for this feature. -->

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Documentation

You can find detailed documentation about AG2 [here](https://docs.ag2.ai/).

In addition, you can find:

- [Research](https://docs.ag2.ai/docs/Research), [blogposts](https://docs.ag2.ai/blog) around AG2, and [Transparency FAQs](https://github.com/ag2ai/ag2/blob/main/TRANSPARENCY_FAQS.md)

- [Discord](https://discord.gg/pAbnFJrkgZ)

- [Contributing guide](https://docs.ag2.ai/docs/contributor-guide/contributing)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## CookBook

Explore detailed implementations with sample code and applications to help you get started with AG2.
[Cookbook](https://github.com/ag2ai/build-with-ag2)


## Related Papers

[AutoGen](https://arxiv.org/abs/2308.08155)

```
@inproceedings{wu2023autogen,
      title={AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework},
      author={Qingyun Wu and Gagan Bansal and Jieyu Zhang and Yiran Wu and Beibin Li and Erkang Zhu and Li Jiang and Xiaoyun Zhang and Shaokun Zhang and Jiale Liu and Ahmed Hassan Awadallah and Ryen W White and Doug Burger and Chi Wang},
      year={2023},
      eprint={2308.08155},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

[EcoOptiGen](https://arxiv.org/abs/2303.04673)

```
@inproceedings{wang2023EcoOptiGen,
    title={Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference},
    author={Chi Wang and Susan Xueqing Liu and Ahmed H. Awadallah},
    year={2023},
    booktitle={AutoML'23},
}
```

[MathChat](https://arxiv.org/abs/2306.01337)

```
@inproceedings{wu2023empirical,
    title={An Empirical Study on Challenging Math Problem Solving with GPT-4},
    author={Yiran Wu and Feiran Jia and Shaokun Zhang and Hangyu Li and Erkang Zhu and Yue Wang and Yin Tat Lee and Richard Peng and Qingyun Wu and Chi Wang},
    year={2023},
    booktitle={ArXiv preprint arXiv:2306.01337},
}
```

[AgentOptimizer](https://arxiv.org/pdf/2402.11359)

```
@article{zhang2024training,
  title={Training Language Model Agents without Modifying Language Models},
  author={Zhang, Shaokun and Zhang, Jieyu and Liu, Jiale and Song, Linxin and Wang, Chi and Krishna, Ranjay and Wu, Qingyun},
  journal={ICML'24},
  year={2024}
}
```

[StateFlow](https://arxiv.org/abs/2403.11322)
```
@article{wu2024stateflow,
  title={StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows},
  author={Wu, Yiran and Yue, Tianwei and Zhang, Shaokun and Wang, Chi and Wu, Qingyun},
  journal={arXiv preprint arXiv:2403.11322},
  year={2024}
}
```

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## Contributors Wall
<a href="https://github.com/ag2ai/ag2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/ag2&max=204" />
</a>

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
  <a href="#readme-top" style="text-decoration: none; color: blue; font-weight: bold;">
    ↑ Back to Top ↑
  </a>
</p>

## License
This project is licensed under the [Apache License, Version 2.0 (Apache-2.0)](./LICENSE).

This project is a spin-off of [AutoGen](https://github.com/microsoft/autogen) and contains code under two licenses:

- The original code from https://github.com/microsoft/autogen is licensed under the MIT License.  See the [LICENSE_original_MIT](./license_original/LICENSE_original_MIT) file for details.

- Modifications and additions made in this fork are licensed under the Apache License, Version 2.0. See the  [LICENSE](./LICENSE) file for the full license text.

We have documented these changes for clarity and to ensure transparency with our user and contributor community. For more details, please see the [NOTICE](./NOTICE.md) file.
