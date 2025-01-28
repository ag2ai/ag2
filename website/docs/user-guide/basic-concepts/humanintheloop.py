from autogen import ConversableAgent

llm_config = {"model": "gpt-4o-mini"}

planner_system_message = """You are a classroom lesson agent.
Given a topic, write a lesson plan for a fourth grade class.
Use the following format:
<title>Lesson plan title</title>
<learning_objectives>Key learning objectives</learning_objectives>
<script>How to introduce the topic to the kids</script>
"""

my_agent = ConversableAgent(
    name="lesson_agent",
    llm_config=llm_config,
    system_message=planner_system_message,
)

# 1. Create our "human" agent
the_human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",
)

# 2. Initiate our chat between the agents
the_human.initiate_chat(recipient=my_agent, message="Today, let's introduce our kids to the solar system.")
