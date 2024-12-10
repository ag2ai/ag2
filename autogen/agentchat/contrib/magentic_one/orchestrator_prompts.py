ORCHESTRATOR_SYSTEM_MESSAGE = """You are an expert orchestrator that coordinates a team of AI agents to accomplish tasks efficiently. 
You excel at breaking down complex tasks, assigning work to the most appropriate agents, and tracking progress."""

ORCHESTRATOR_CLOSED_BOOK_PROMPT = """I will present you with a task to coordinate. Before we begin, please analyze the request 
and categorize the relevant information into the following sections:

Task to analyze:
{task}

Please categorize the information as follows:

1. GIVEN OR VERIFIED FACTS
   - List any specific facts, requirements, or constraints explicitly stated in the task
   - Include any numerical values, dates, or specific criteria mentioned

2. FACTS TO LOOK UP
   - List information that needs to be researched
   - Specify potential sources where this information might be found
   - Note any technical terms or concepts that need verification

3. FACTS TO DERIVE
   - List any conclusions that need to be drawn through analysis
   - Identify any calculations or logical deductions needed
   - Note any relationships between different pieces of information

4. EDUCATED GUESSES
   - List any reasonable assumptions that can be made
   - Note any industry standards or common practices that might apply
   - Include any contextual insights based on the nature of the task

Please provide only these four sections with their relevant points. Do not include any additional commentary or planning at this stage."""

ORCHESTRATOR_PLAN_PROMPT = """Based on our analysis, we will work with the following team members:

{team_description}

Given the team's capabilities and the task requirements, please create a structured plan that:
1. Leverages each agent's strengths
2. Addresses the identified information gaps
3. Ensures efficient task completion

Please provide the plan in bullet points, with clear assignments for each step. Remember:
- Only include available team members
- Consider dependencies between tasks
- Account for potential failure points

Your plan should be concise and actionable."""

ORCHESTRATOR_LEDGER_PROMPT = """Please evaluate our current progress on this task:

Original Task:
{task}

Available Team:
{team_description}

Based on our current state, please provide a JSON response with the following structure:

{{"is_request_satisfied": {{"reason": "Explain why the task is or is not complete", "answer": boolean}}, "is_in_loop": {{"reason": "Explain if we're repeating actions without progress", "answer": boolean}}, "is_progress_being_made": {{"reason": "Explain if we're moving toward our goal", "answer": boolean}}, "next_speaker": {{"reason": "Explain why this agent should act next", "answer": "Agent role name"}}, "instruction_or_question": {{"reason": "Explain why this is the appropriate next action. Include all information the agent needs to execute the action", "answer": "Specific instruction for the next agent"}}}}

Ensure your response is valid JSON and uses the exact agent role names: {agent_roles}"""

ORCHESTRATOR_UPDATE_FACTS_PROMPT = """We need to reassess our understanding of the task:

Original Task:
{task}

Previous Fact Sheet:
{previous_facts}

Please provide an updated fact sheet that:
1. Incorporates new information we've gathered
2. Revises any incorrect assumptions
3. Adds new educated guesses based on what we've learned
4. Moves verified guesses to the facts section

Use the same four-section format as before:
1. GIVEN OR VERIFIED FACTS
2. FACTS TO LOOK UP
3. FACTS TO DERIVE
4. EDUCATED GUESSES

Explain any significant changes you make to the fact sheet."""

ORCHESTRATOR_UPDATE_PLAN_PROMPT = """We need to revise our approach based on recent challenges:

Team Composition:
{team_description}

Please provide:
1. A brief analysis of what went wrong in our previous attempt
2. A revised plan that:
   - Addresses the identified issues
   - Avoids previous pitfalls
   - Includes specific guidance for the team
   
Present the new plan in bullet points, ensuring each step has a clear owner and success criteria."""

ORCHESTRATOR_GET_FINAL_ANSWER = """Please provide a final response for this task:

Original Task:
{task}

Your response should:
1. Directly address the original request
2. Summarize key findings or outcomes
3. Present the information in a clear, user-friendly format
4. Include any important caveats or limitations

Format your response as if speaking directly to the user who made the request."""

ORCHESTRATOR_REPLAN_PROMPT = """We need to create a new plan due to {reason}:

Current Team:
{team_description}

Task Status:
- Completed steps: {completed_steps}
- Failed attempts: {failed_attempts}
- Current blockers: {blockers}

Please provide:
1. A revised approach that addresses our current challenges
2. Specific steps to overcome identified blockers
3. Alternative strategies if available

Present the new plan in bullet points, with clear ownership and success criteria for each step.""" 


ORCHESTRATOR_SYNTHESIZE_PROMPT = """
We are working to address the following user request:

{task}


To answer this request we have assembled the following team:

{team}


Here is an initial fact sheet to consider:

{facts}


Here is the plan to follow as best as possible:

{plan}
"""