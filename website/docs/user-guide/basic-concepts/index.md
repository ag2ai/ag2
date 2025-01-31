*This page is no longer necessary*

Replaced by these pages, in order:
- installing-ag2.md
- llm-configuration.md
- conversable-agent.md
- human-in-the-loop.md
- orchestrations.md
- groupchat.md
- swarm.md
- tools.md
- structured-outputs.md
- ending-a-chat.md



-- BELOW CONTENT MOVED INTO MD FILES --

In this guide, you'll learn how to create AG2 agents and get them to work together.

## Before we get started...

### GitHub Codespaces

The fastest way to get started is to open the code snippets in GitHub Codespaces and start playing with the code yourself. Each snippet has a button that will open it in a fully functioning AG2 Codespace.

Add an OPENAI_API_KEY secret to your GitHub Codespaces so your agents can use an LLM. See the [instructions here](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces).

### Installing AG2

Alternatively, if you'd like to install AG2 on your machine:

```bash
pip install ag2
```

:::note
We recommended using a virtual environment
:::

### LLM configurations

AG2 agents can use LLMs through providers such as OpenAI, Anthropic, Google, Amazon, Mistral AI, Cerebras, and Groq. Locally hosted models can also be used through Ollama, LiteLLM, and LM Studio.

The examples include an LLM configuration for OpenAI's `GPT-4o mini` model. Set your `OPENAI_API_KEY` environment variable accordingly:

macOS / Linux
```bash
export OPENAI_API_KEY="your_api_key_here"
```

 Windows
 ```bash
setx OPENAI_API_KEY "your_api_key_here"
 ```

If you would like to use a different provider, [see the model providers list](/docs/user-guide/models/).


# Say hello to ConversableAgent

ConversableAgent is at the heart of all AG2 agents while also being a fully functioning agent.

Let's *converse* with ConversableAgent in just 4 simple steps.

--SNIPPET: conversableagentchat.py

```python
# TEMPORARY, THIS WILL BE REPLACED BY ABOVE SNIPPET

# 1. Import our agent class
from autogen import ConversableAgent

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    Provider defaults to OpenAI and uses the OPENAI_API_KEY environment variable
llm_config = {"model": "gpt-4o-mini"}

# 3. Create our agent
my_agent = ConversableAgent(
    name="helpful_agent",
    llm_config=llm_config,
    system_message="You are a poetic AI assistant",
)

# 4. Chat directly with our agent
my_agent.run("In one sentence, what's the big deal about AI?")
```

Let's break it down:

1. Import `ConversableAgent`, you'll find the most popular classes available directly from `autogen`.

2. Create our LLM configuration to define the LLM that our agent will use.

3. Create our ConversableAgent give them a unique name, and use `system_message` to define their purpose.

4. Chat with the agent using their `run` method, passing in our starting message.

```console
human (to helpful_agent):

In one sentence, what's the big deal about AI?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
helpful_agent (to human):

AI’s a marvel that transforms the mundane,
Empowering minds, making tasks less a strain.

--------------------------------------------------------------------------------
```

# Human in the loop

If you run the previous example you'll be able to chat with the agent and this is because another agent was automatically created to represent you, the human in the loop, when using an agent's `run` method.

As you build your own workflows, you'll want to create your own *human in the loop* agents and decide if and how to use them. To do so, simply use the ConversableAgent and set the `human_input_mode` to `ALWAYS`.

Let's start to build a more useful scenario, a classroom lesson planner, and create our human agent.

--SNIPPET: humanintheloop.py

```python
# TEMPORARY, THIS WILL BE REPLACED BY ABOVE SNIPPET

from autogen import ConversableAgent
llm_config = {"model": "gpt-4o-mini"}

planner_system_message = """You are a classroom lesson agent.
Given a topic, write a lesson plan for a fourth grade class.
Use the following format:
<title>Lesson plan title</title>
<learning_objectives>Key learning objectives</learning_objectives>
<script>How to introduce the topic to the kids</script>
"""

lesson_planner = ConversableAgent(
    name="lesson_agent",
    llm_config=llm_config,
    system_message=planner_system_message,
)

# 1. Create our human input agent
the_human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",
)

# 2. Initiate our chat between the agents
the_human.initiate_chat(recipient=lesson_planner, message="Today, let's introduce our kids to the solar system.")
```

1. Create a second agent and set its `human_input_mode` with no `llm_config` required.

2. Our `the_human` agent starts a conversation by sending a message to `lesson_planner`. An agent's `initiate_chat` method is used to start a conversation between two agents.

# Many agents

Many hands make for light work, so it's time to move on from our simple two-agent conversations and think about orchestrating workflows containing many agents, a strength of the AG2 framework.

There are two mechanisms for building multi-agent workflows, GroupChat and Swarm.

GroupChat contains a number of built-in conversation patterns to determine the next agent. You can also define your own.

Swarm is a conversation pattern based on agents with handoffs. There's a shared context and each agent has tools and the ability to transfer control to other agents. The [original swarm concept](https://github.com/openai/swarm) was created by OpenAI.

:::note
We'll refer to *[conversation patterns](https://docs.ag2.ai/docs/tutorial/conversation-patterns)* throughout the documentation - they are simply a structured way of organizing the flow between agents.
:::

## GroupChat

`GroupChat` has four built-in conversation patterns:

| Method | Agent selection|
| --- | --- |
| `auto` (default) | Automatic, chosen by the `GroupChatManager` using an LLM |
| `round_robin` | Sequentially in their added order |
| `random` | Randomly |
| `manual` | Selected by you at each turn |
| *Callable* | Create your own flow |

Coordinating the `GroupChat` is the `GroupChatManager`, an agent that provides a way to start and resume multi-agent chats.

Let's enhance our lesson planner example to include a lesson reviewer and a teacher agent.

:::tip
You can start any multi-agent chats using the `initiate_chat` method
:::

--SNIPPET: groupchat.py

```python
# TEMPORARY, THIS WILL BE REPLACED BY ABOVE SNIPPET

from autogen import ConversableAgent, GroupChat, GroupChatManager
llm_config = {"model": "gpt-4o-mini"}

planner_message = """You are a classroom lesson agent.
Given a topic, write a lesson plan for a fourth grade class.
Use the following format:
<title>Lesson plan title</title>
<learning_objectives>Key learning objectives</learning_objectives>
<script>How to introduce the topic to the kids</script>
"""

# 1. Add a separate 'description' for our planner and reviewer agents
planner_description = "Creates or revises lesson plans."

lesson_planner = ConversableAgent(
    name="planner_agent",
    llm_config=llm_config,
    system_message=planner_message,
    description=planner_description,
)

reviewer_message = """You are a classroom lesson reviewer.
You compare the lesson plan to the fourth grade curriculum and provide a maximum of 3 recommended changes.
Provide only one round of reviews to a lesson plan.
"""

reviewer_description = """Provides one round of reviews to a lesson plan
for the lesson_planner to revise."""

lesson_reviewer = ConversableAgent(
    name="reviewer_agent",
    llm_config=llm_config,
    system_message=reviewer_message,
    description=reviewer_description,
)

# 2. The teacher's system message can also be used as a description, so we don't define it
teacher_message = """You are a classroom teacher.
You decide topics for lessons and work with a lesson planner.
and reviewer to create and finalise lesson plans.
When you are happy with a lesson plan, output "DONE!".
"""

teacher = ConversableAgent(
    name="teacher_agent",
    llm_config=llm_config,
    system_message=teacher_message,
    # 3. Our teacher can end the conversation by saying DONE!
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
)

# 4. Create the GroupChat with agents and selection method
groupchat = GroupChat(
    agents=[teacher, lesson_planner, lesson_reviewer],
    speaker_selection_method="auto",
    messages=[],
)

# 5. Our GroupChatManager will manage the conversation and uses an LLM to select the next agent
manager = GroupChatManager(
    name="group_manager",
    groupchat=groupchat,
    llm_config=llm_config,
)

# 6. Starting a chat with the GroupChatManager as the recipient kicks off the group chat
teacher.initiate_chat(recipient=manager, message="Today, let's introduce our kids to the solar system.")
```
1. Separate to `system_message`, we add a `description` for our planner and reviewer agents and this is used exclusively for the purposes of determining the next agent by the `GroupChatManager` when using automatic speaker selection.

2. The teacher's `system_message` is suitable as a description so, by not setting it, the `GroupChatManager` will use the `system_message` for the teacher when determining the next agent.

3. The workflow is ended when the teacher's message contains the phrase "DONE!".

4. Construct the `GroupChat` with our agents and selection method as automatic (which is the default).

5. `GroupChat` requires a `GroupChatManager` to manage the chat and an LLM configuration is needed because they'll use an LLM to decide the next agent.

6. Starting a chat with the `GroupChatManager` as the `recipient` kicks off the group chat.

```console
teacher_agent (to group_manager):

Today, let's introduce our kids to the solar system.

--------------------------------------------------------------------------------

Next speaker: planner_agent


>>>>>>>> USING AUTO REPLY...
planner_agent (to group_manager):

<title>Exploring the Solar System</title>
<learning_objectives>
1. Identify and name the planets in the solar system.
2. Describe key characteristics of each planet.
3. Understand the concept of orbit and how planets revolve around the sun.
4. Develop an appreciation for the scale and structure of our solar system.
</learning_objectives>
<script>
"Good morning, class! Today, we are going to embark on an exciting journey through our solar system. Have any of you ever looked up at the night sky and wondered what those bright dots are? Well, those dots are often stars, but some of them are planets in our own solar system!

To start our adventure, I want you to close your eyes and imagine standing on a giant spaceship, ready to zoom past the sun. Does anyone know what the sun is? (Pause for responses.) Right! The sun is a star at the center of our solar system.

Today, we are going to learn about the planets that travel around the sun - but not just their names, we're going to explore what makes each of them special! We will create a model of the solar system together, and by the end of the lesson, you will be able to name all the planets and tell me something interesting about each one.

So, are you ready to blast off and discover the wonders of space? Let's begin!"
</script>

--------------------------------------------------------------------------------

Next speaker: reviewer_agent


>>>>>>>> USING AUTO REPLY...
reviewer_agent (to group_manager):

**Review of the Lesson Plan: Exploring the Solar System**

1. **Alignment with Curriculum Standards**: Ensure that the lesson includes specific references to the fourth grade science standards for the solar system. This could include discussing gravity, the differences between inner and outer planets, and the role of the sun as a stable center of our solar system. Adding this information will support a deeper understanding of the topic and ensure compliance with state educational standards.

2. **Interactive Activities**: While creating a model of the solar system is a great hands-on activity, consider including additional interactive elements such as a group discussion or a game that reinforces the learning objectives. For instance, incorporating a "planet facts" game where students can share interesting facts about each planet would further engage the students and foster collaborative learning.

3. **Assessment of Learning**: It would be beneficial to include a formative assessment to gauge students' understanding at the end of the lesson. This could be a quick quiz, a group presentation about one planet, or a drawing activity where students depict their favorite planet and share one fact about it. This will help reinforce the learning objectives and provide students with an opportunity to demonstrate their knowledge.

Making these adjustments will enhance the educational experience and align it more closely with fourth-grade learning goals.

--------------------------------------------------------------------------------

Next speaker: planner_agent


>>>>>>>> USING AUTO REPLY...
planner_agent (to group_manager):

**Revised Lesson Plan: Exploring the Solar System**

<title>Exploring the Solar System</title>
<learning_objectives>
1. Identify and name the planets in the solar system according to grade-level science standards.
2. Describe key characteristics of each planet, including differences between inner and outer planets.
3. Understand the concept of orbit and how gravity keeps planets revolving around the sun.
4. Develop an appreciation for the scale and structure of our solar system and the sun's role as the center.
</learning_objectives>
<script>
"Good morning, class! Today, we are going to embark on an exciting journey through our solar system. Have any of you ever looked up at the night sky and wondered what those bright dots are? Well, those dots are often stars, but some of them are planets in our own solar system!

To start our adventure, I want you to close your eyes and imagine standing on a giant spaceship, ready to zoom past the sun. Does anyone know what the sun is? (Pause for responses.) That's right! The sun is a star at the center of our solar system.

Now, today's goal is not only to learn the names of the planets but also to explore what makes each of them unique. We'll create a model of the solar system together, and through this process, we will also talk about the differences between the inner and outer planets.

As part of our exploration, we'll play a fun "planet facts" game. After learning about the planets, I will divide you into small groups, and each group will get a planet to research. You’ll find interesting facts about the planet, and we will come together to share what you discovered!

At the end of our lesson, I'll give you a quick quiz to see how much you've learned about the planets, or you can draw your favorite planet and share one cool fact you found with the class.

So, are you ready to blast off and discover the wonders of space? Let's begin!"
</script>

**Interactive Activities**:
- **Planet Facts Game**: After discussing each planet, students will work in groups to find and share a unique fact about their assigned planet.

**Assessment of Learning**:
- **Individual Reflection Drawing**: Students draw their favorite planet and write one fact about it.
- **Quick Quiz**: A short quiz at the end to assess understanding of the planets and their characteristics.

This revised plan incorporates additional interactive elements and assessments that align with educational standards and enhance the overall learning experience.

--------------------------------------------------------------------------------

Next speaker: teacher_agent


>>>>>>>> USING AUTO REPLY...
teacher_agent (to group_manager):

DONE!

--------------------------------------------------------------------------------
```

## Swarm

Swarms provide controllable flows between agents that are determined at the agent-level. You define hand-off, post-tool, and post-work transitions from an agent to another agent (or to end the swarm).

When designing your swarm, think about your agents in a diagram with the lines between agents being your hand-offs. Each line will have a condition statement which an LLM will evaluate. Control stays with an agent while they execute their tools and once they've finished with their tools the conditions to transition will be evaluated.

One of the unique aspects of a swarm is a shared context. ConversableAgents have a context dictionary but in a swarm that context is made common across all agents, allowing a state of the workflow to be maintained and viewed by all agents. This context can also be used within the hand off condition statements, providing more control of transitions.

AG2's swarm has a number of unique capabilities, find out more in our [Swarm documentation](https://docs.ag2.ai/docs/topics/swarm).

Here's our lesson planner workflow using AG2's Swarm.

--SNIPPET: swarm.py

```python
# TEMPORARY, THIS WILL BE REPLACED BY ABOVE SNIPPET

from autogen import SwarmAgent, initiate_swarm_chat, AfterWorkOption, ON_CONDITION, AFTER_WORK, SwarmResult

llm_config = {"model": "gpt-4o-mini", "cache_seed": None}

# 1. Context
shared_context = {
    "lesson_plans": [],
    "lesson_reviews": [],
    # Will be decremented, resulting in 0 (aka False) when no reviews are left
    "reviews_left": 2,
}

# 2. Functions
def record_plan(lesson_plan: str, context_variables: dict) -> SwarmResult:
    """Record the lesson plan"""
    context_variables["lesson_plans"].append(lesson_plan)

    # Returning the updated context so the shared context can be updated
    return SwarmResult(context_variables=context_variables)

def record_review(lesson_review: str, context_variables: dict) -> SwarmResult:
    """After a review has been made, increment the count of reviews"""
    context_variables["lesson_reviews"].append(lesson_review)
    context_variables["reviews_left"] -= 1

    # Controlling the flow to the next agent from a tool call
    return SwarmResult(
        agent=teacher if context_variables["reviews_left"] < 0 else lesson_planner,
        context_variables=context_variables
    )

planner_message = """You are a classroom lesson planner.
Given a topic, write a lesson plan for a fourth grade class.
If you are given revision feedback, update your lesson plan and record it.
Use the following format:
<title>Lesson plan title</title>
<learning_objectives>Key learning objectives</learning_objectives>
<script>How to introduce the topic to the kids</script>
"""

# 3. Our agents now have tools to use (functions above)
lesson_planner = SwarmAgent(
    name="planner_agent",
    llm_config=llm_config,
    system_message=planner_message,
    functions=[record_plan]
)

reviewer_message = """You are a classroom lesson reviewer.
You compare the lesson plan to the fourth grade curriculum
and provide a maximum of 3 recommended changes for each review.
Make sure you provide recommendations each time the plan is updated.
"""

lesson_reviewer = SwarmAgent(
    name="reviewer_agent",
    llm_config=llm_config,
    system_message=reviewer_message,
    functions=[record_review]
)

teacher_message = """You are a classroom teacher.
You decide topics for lessons and work with a lesson planner.
and reviewer to create and finalise lesson plans.
"""

teacher = SwarmAgent(
    name="teacher_agent",
    llm_config=llm_config,
    system_message=teacher_message,
)

# 4. Transitions using hand-offs

# Lesson planner will create a plan and hand off to the reviewer if we're still
# allowing reviews. After that's done, transition to the teacher.
lesson_planner.register_hand_off(
    [
        ON_CONDITION(
            target=lesson_reviewer,
            condition="After creating/updating and recording the plan, it must be reviewed.",
            available="reviews_left"),
        AFTER_WORK(
            agent=teacher
        )
    ]
)

# Lesson reviewer will review the plan and return control to the planner. If they don't
# provide feedback, they'll revert back to the teacher.
lesson_reviewer.register_hand_off(
    [
        ON_CONDITION(
            target=lesson_planner,
            condition="After new feedback has been made and recorded, the plan must be updated."
        ),
        AFTER_WORK(
            agent=teacher
        )
    ]
)

# Teacher works with the lesson planner to create a plan. When control returns to them and
# a plan exists, they'll end the swarm.
teacher.register_hand_off(
    [
        ON_CONDITION(
            target=lesson_planner,
            condition="Create a lesson plan.",
            available="reviews_left"
        ),
        AFTER_WORK(AfterWorkOption.TERMINATE)
    ]
)

# 5. Run the Swarm which returns the chat and updated context variables
chat_result, context_variables, last_agent = initiate_swarm_chat(
    initial_agent=teacher,
    agents=[lesson_planner, lesson_reviewer, teacher],
    messages="Today, let's introduce our kids to the solar system.",
    context_variables=shared_context,
)

print(f"Number of reviews: {len(context_variables['lesson_reviews'])}")
print(f"Reviews remaining: {context_variables['reviews_left']}")
print(f"Final Lesson Plan:\n{context_variables['lesson_plans'][-1]}")
```
1. Our shared context, available in function calls and on agents.

2. Functions that represent the work the agents carry out, these the update shared context and, optionally, managed transitions.

3. Agents setup with their tools, `functions`, and a system message and LLM configuration.

4. The important hand-offs, defining the conditions for which to transfer to other agents and what to do after their work is finished (equivalent to no longer calling tools). Transfer conditions can be turned on/off using the `available` parameter.

5. Kick off the swarm with our agents and shared context. Similar to `initiate_chat`, `initiate_swarm_chat` returns the chat result (messages and summary) and the final shared context.

```console
Number of reviews: 2
Reviews remaining: 0
Final Lesson Plan:
<title>Exploring the Solar System</title>
<learning_objectives>Students will be able to identify and describe the planets in the Solar System, understand the concept of orbits, and recognize the sun as the center of our Solar System.</learning_objectives>
<script>Using a large poster of the Solar System, I will introduce the planets one by one, discussing key characteristics such as size, color, and distance from the sun. I will engage the students by asking questions about their favorite planets and what they know about them. We will do a quick demo using a simple model to show how planets orbit around the sun. Students will create their own solar system models in small groups with various materials to understand the scale and distance better, fostering teamwork. We will incorporate a multimedia presentation to visualize the orbits and relative sizes of the planets. Finally, a short assessment will be conducted at the end of the lesson to gauge students' understanding, using quizzes or presentations of their models.</script>
```

# Tools

Agents gain significant utility through tools as they provide access to external data, APIs, and functionality.

In AG2, using tools is done in two parts, an agent *suggests* which tools to use (via their LLM) and another *executes* the tool.

::note
In a conversation the executor agent must follow the agent that suggests a tool.
::

In the swarm example above, we attached tools to our agents and, as part of the swarm, AG2 created a tool executor agent to run recommended tools. Typically, you'll create two agents, one to decide which tool to use and another to execute it.

--SNIPPET: toolregister.py

```python
# TEMPORARY, THIS WILL BE REPLACED BY ABOVE SNIPPET

from autogen import ConversableAgent, register_function
from typing import Annotated
from datetime import datetime
llm_config = {"model": "gpt-4o-mini"}

# 1. Our tool, returns the day of the week for a given date
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, '%Y-%m-%d')
    return date.strftime('%A')

# 2. Agent for determining whether to run the tool
date_agent = ConversableAgent(
    name="date_agent",
    system_message="You get the day of the week for a given date.",
    llm_config=llm_config,
)

# 3. And an agent for executing the tool
executor_agent = ConversableAgent(
    name="executor_agent",
    human_input_mode="NEVER",
)

# 4. Registers the tool with the agents, the description will be used by the LLM
register_function(
    get_weekday,
    caller=date_agent,
    executor=executor_agent,
    description="Get the day of the week for a given date",
)

# 5. Two-way chat ensures the executor agent follows the suggesting agent
chat_result = executor_agent.initiate_chat(
    recipient=date_agent,
    message="I was born on the 25th of March 1995, what day was it?",
    max_turns=2,
)

print(chat_result.chat_history[-1]["content"])
```
1. Here's the tool, a function, that we'll attach to our agents, the `Annotated` parameter will be  included in the call to the LLM so it understands what the `date_string` needs.

2. The date_agent will determine whether to use the tool, using its LLM.

3. The executor_agent will run the tool and return the output as its reply.

4. Registering the tool with the agents and giving it a description to help the LLM determine.

5. We have a two-way chat, so after the date_agent the executor_agent will run and, if it sees that the date_agent suggested the use of the tool, it will execute it.

```console
executor_agent (to date_agent):

I was born on the 25th of March 1995, what day was it?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
date_agent (to executor_agent):

***** Suggested tool call (call_iOOZMTCoIVVwMkkSVu04Krj8): get_weekday *****
Arguments:
{"date_string":"1995-03-25"}
****************************************************************************

--------------------------------------------------------------------------------

>>>>>>>> EXECUTING FUNCTION get_weekday...
Call ID: call_iOOZMTCoIVVwMkkSVu04Krj8
Input arguments: {'date_string': '1995-03-25'}
executor_agent (to date_agent):

***** Response from calling tool (call_iOOZMTCoIVVwMkkSVu04Krj8) *****
Saturday
**********************************************************************

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
date_agent (to executor_agent):

It was a Saturday.

--------------------------------------------------------------------------------
```

Alternatively, you can use decorators to register a tool. So, instead of using `register_function`, you can register them with the function definition.
```python
@date_agent.register_for_llm(description="Get the day of the week for a given date")
@executor_agent.register_for_execution()
def get_weekday(date_string: Annotated[str, "Format: YYYY-MM-DD"]) -> str:
    date = datetime.strptime(date_string, '%Y-%m-%d')
    return date.strftime('%A')
```

# Structured outputs

Working with freefrom text from LLMs isn't optimal when you know how you want the reply formatted.

Using structured outputs, you define a class, based on Pydantic's `BaseModel`, for the reply format you want and attach it to the LLM configuration. Replies from agent using that configuration will be in a matching JSON format.

In earlier examples, we created a classroom lesson plan and provided guidance in the agent's system message to put the content in tags, like `<title>` and `<learning_objectives>`. Using structured outputs we can ensure that our lesson plans are formatted.

--SNIPPET: structured_output.py
```python
# TEMPORARY, THIS WILL BE REPLACED BY ABOVE SNIPPET

from autogen import ConversableAgent
from pydantic import BaseModel
import json

# 1. Define our lesson plan structure, a lesson with a number of objectives
class LearningObjective(BaseModel):
    title: str
    description: str

class LessonPlan(BaseModel):
    title: str
    learning_objectives: list[LearningObjective]
    script: str

# 2. Add our lesson plan structure to the LLM configuration
llm_config = {
    "model": "gpt-4o-mini",
    "response_format": LessonPlan,
}

# 3. The agent's system message doesn't need any formatting instructions
system_message = """You are a classroom lesson agent.
Given a topic, write a lesson plan for a fourth grade class.
"""

my_agent = ConversableAgent(name="lesson_agent", llm_config=llm_config, system_message=system_message)

# 4. Chat directly with our agent
chat_result = my_agent.run("In one sentence, what's the big deal about AI?")

# 5. Get and print our lesson plan
lesson_plan_json = json.loads(chat_result.chat_history[-1]["content"])
print(json.dumps(lesson_plan_json, indent=2))
```


```console
{
  "title": "Exploring the Solar System",
  "learning_objectives": [
    {
      "title": "Understanding the Solar System",
      "description": "Students will learn the names and order of the planets in the solar system."
    },
    {
      "title": "Identifying Planet Characteristics",
      "description": "Students will be able to describe at least one distinctive feature of each planet."
    },
    {
      "title": "Creating a Planet Fact Sheet",
      "description": "Students will create a fact sheet for one planet of their choice."
    }
  ],
  "script": "Introduction (10 minutes):\nBegin the class by asking students what they already know about the solar system. Write their responses on the board. \n\nIntroduce the topic by explaining that today they will be learning about the solar system, which includes the Sun, planets, moons, and other celestial objects.\n\nDirect Instruction (20 minutes):\nUse a visual aid (such as a poster or video) to show the solar system's structure. \n\nDiscuss the eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. \n\nFor each planet, mention:\n- Its position from the Sun.\n- Key characteristics (e.g., size, color, temperature).\n- Any notable features (e.g., rings, atmosphere). \n\nInteractive Activity (15 minutes):\nSplit the class into small groups. Give each group a set of planet cards that include pictures and information. Have them work together to put the planets in order from the Sun. Each group will present their order and one interesting fact about each planet they discussed."
}
```
::tip
Add a `format` function to the LessonPlan class in the example to convert the returned value into a string. [Example here](/docs/use-cases/notebooks/notebooks/agentchat_structured_outputs#define-the-reasoning-model-2).
::

# Ending a chat

There are a number of ways a chat can end:
1. The maximum number of turns in a chat is reached
2. An agent's termination function passes on a received message
3. An agent automatically replies a maximum number of times
4. A human replies with 'exit' when prompted
5. In a group chat, there's no next agent
6. In a swarm, transitioning to AfterWorkOption.TERMINATE
7. Custom reply functions

### 1. Maximum turns
For GroupChat and swarm, use `max_round` to set a limit on the number of replies. Each round represents an agent speaking and includes the initial message.

```python
# GroupChat with a maximum of 5 rounds
groupchat = GroupChat(
    agents=[agent_a, agent_b, agent_c],
    speaker_selection_method="round_robin",
    max_round=5,
    ...
)
gcm = GroupChatManager(
    ...
)
agent_a.initiate_chat(gcm, "first message")
# 1. agent_a with "first message" > 2. agent_b > 3. agent_c > 4. agent_a > 5. agent_b > end

# Swarm with a maximum of 5 rounds
initiate_swarm_chat(
    agents=[agent_a, agent_b, agent_c],
    max_round=5,
    messages="first message"
    ...
)
# When initial agent is set to agent_a and agents hand off to the next agent.
# 1. User with "first message" > 2. agent_a > 3. agent_b > 4. agent_c > 5. agent_a > end
```

For `initiate_chat` use `max_turns`. Each turn represents a round trip of both agents speaking and includes the initial message.

```python
# initiate_chat with a maximum of 2 turns across the 2 agents (effectively 4 steps)
agent_a.initiate_chat(
    recipient=agent_b,
    max_turns=2,
    message="first message"
)
# 1. agent_a with "first message" > 1. agent_b > 2. agent_a > 2. agent_b > end
```

### 2. Terminating message
Agents can check their received message for a termination condition and, if that condition returns `True`, the chat will be ended. This check is carried out before they reply.

When constructing an agent, use the `is_termination_msg` parameter with a Callable. To save creating a function, you can use a [lambda function](https://docs.python.org/2/tutorial/controlflow.html#lambda-expressions) as in the example below.

It's important to put the termination check on the agents that will receive the message, not the agent creating the message.

```python
agent_a = ConversableAgent(
    system_message="You're a helpful AI assistant, end your responses with 'DONE!'"
    ...
)

# Terminates when the agent receives a message with "DONE!" in it.
agent_b = ConversableAgent(
    is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper()
    ...
)

# agent_b > agent_a replies with message "... DONE!" > agent_b ends before replying
```

::note
If the termination condition is met and the agent's `human_input_mode` is "ALWAYS" or 'TERMINATE' (ConversableAgent's default), you will be asked for input and can decide to end the chat. If it is "NEVER" it will end immediately.
::

### 3. Number of automatic replies

A conversation can be ended when an agent has responded to another agent a certain number of times. An agent evaluates this when it is next their time to reply, not immediately after they have replied.

When constructing an agent, use the `max_consecutive_auto_reply` parameter to set this.

```python
agent_a = ConversableAgent(
    max_consecutive_auto_reply=2
    ...
)

agent_b = ConversableAgent(
    ...
)

agent_a.initiate_chat(agent_b, ...)

# agent_a > agent_b > agent_a with first auto reply > agent_b > agent_a with second auto reply > agent_b > agent_a ends before replying
```

::note
If the agent's `human_input_mode` is "ALWAYS" or 'TERMINATE' (ConversableAgent's default), you will be asked for input and can decide to end the chat. If it is "NEVER" it will end immediately.
::

### 4. Human replies with 'exit'
During the course of the conversation, if you are prompted and reply 'exit', the chat will end.

```console
--------------------------------------------------------------------------------
Please give feedback to agent_a. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: exit
```

### 5. GroupChat, no next agent
If the next agent in a GroupChat can't be determined the chat will end.

If you are customizing the speaker selection method with a Callable, return `None` to end the chat.

### 6. Swarm, transitioning to end the chat
In a swarm, if you transition to `AfterWorkOption.TERMINATE` it will end the swarm. The default swarm-level AfterWork option is `AfterWorkOption.TERMINATE` and this will apply to any agent in the swarm that doesn't have an AfterWork hand-off specified.

Additionally, if you transition to `AfterWorkOption.REVERT_TO_USER` but have not specified a `user_agent` in `initiate_swarm_chat` then it will end the swarm.

### 7. Reply functions
AG2 provides the ability to create custom reply functions for an agent using `register_reply`.

In your function, return a `Tuple` of `True, None` to indicate that the reply is final with `None` indicating there's no reply and it should end the chat.

```python
agent_a = ConversableAgent(
    ...
)

agent_b = ConversableAgent(
    ...
)

def my_reply_func(
    recipient: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[Any] = None,
) -> Tuple[bool, Union[str, Dict, None]]:
    return True, None # Indicates termination

# Register the reply function as the agent_a's first reply function
agent_a.register_reply(
    trigger=[Agent, None],
    reply_func=my_reply_func,
    position=0

)

agent_a.initiate_chat(agent_b, ...)

# agent_a > agent_b > agent_a ends with first custom reply function
```

# Code execution
...

# Other topics TBD

# Resuming a chat

# Advanced Agents

- CaptainAgent
- ReasoningAgent
- DocumentAgent
- WebSurferAgent
- DiscordAgent

# Agent Capabilities

# How do I ...
