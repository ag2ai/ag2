from typing import Optional, Union, Literal, Callable                                                                                                                                             
from autogen.agentchat import ConversableAgent                                                                                                                                                         
                                                                                                                                                                                                   
def create_coder_agent(                                                                                                                                                                           
    name: str,                                                                                                                                                                                    
    system_message: Optional[str] = """You are a helpful AI assistant.                                                                                                                            
    Solve tasks using your coding and language skills.                                                                                                                                                
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.                                                            
        1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the    
    current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.              
        2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.                                                             
    Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.                           
    When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can' 
    modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.                                       
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do n 
    ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.                                              
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is  
    not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.                
    When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.                                                                                   
    Reply "TERMINATE" in the end when everything is done.""",                                                                                                                                         
    description: Optional[str] = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills.",                                     
    **kwargs                                                                                                                                                                                      
    ) -> ConversableAgent:                                                                                                                                                                            
    """Create a coder agent that can write and execute code."""                                                                                                                                   
                                                                                                                                                                                                   
    agent = ConversableAgent(                                                                                                                                                                     
        name=name,                                                                                                                                                                                
        system_message=system_message,                                                                                                                                                            
        description=description,                                                                                                                                                                  
        code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },                                                                                                                        
        **kwargs                                                                                                                                                                                  
    )                                                                                                                                                                                             
                                                                                                                                                                                                   
    # Clear existing reply functions and register new ones in specific order                                                                                                                      
    agent._reply_func_list = []                                                                                                                                                                   
                                                                                                                                                                                                
    # Register reply functions in order:                                                                                                                                                          
    # 1. Check for termination/human input                                                                                                                                                        
    # 2. Try code execution                                                                                                                                                                       
    # 3. Try function/tool calls                                                                                                                                                                  
    # 4. Fall back to LLM response                                                                                                                                                                
    agent.register_reply([ConversableAgent, None], agent.check_termination_and_human_reply)                                                                                                       
    agent.register_reply([ConversableAgent, None], agent.generate_code_execution_reply)                                                                                                           
    agent.register_reply([ConversableAgent, None], agent.generate_function_call_reply)                                                                                                            
    agent.register_reply([ConversableAgent, None], agent.generate_tool_calls_reply)                                                                                                               
    agent.register_reply([ConversableAgent, None], agent.generate_oai_reply)                                                                                                                      
                                                                                                                                                                                                
    return agent       