# Custom Code Executor

In this guide we will show you how to create a custom code executor that runs
code inside the same Jupyter notebook as this one.

First, let's install the required dependencies:


```python
! pip -qqq install pyautogen matplotlib yfinance
```


```python
import os
from typing import List

from IPython import get_ipython

from autogen import ConversableAgent
from autogen.coding import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, MarkdownCodeExtractor
```

Now we can create the custom code executor class by subclassing the
`CodeExecutor` protocol and implementing the `execute_code_blocks` method.


```python
class NotebookExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        # Extact code from markdown blocks.
        return MarkdownCodeExtractor()

    def __init__(self) -> None:
        # Get the current IPython instance running in this notebook.
        self._ipython = get_ipython()

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        for code_block in code_blocks:
            result = self._ipython.run_cell("%%capture --no-display cap\n" + code_block.code)
            log += self._ipython.ev("cap.stdout")
            log += self._ipython.ev("cap.stderr")
            if result.result is not None:
                log += str(result.result)
            exitcode = 0 if result.success else 1
            if result.error_before_exec is not None:
                log += f"\n{result.error_before_exec}"
                exitcode = 1
            if result.error_in_exec is not None:
                log += f"\n{result.error_in_exec}"
                exitcode = 1
            if exitcode != 0:
                break
        return CodeResult(exit_code=exitcode, output=log)
```

Now we can use the new custom code executor in our agents.


```python
code_writer_agent = ConversableAgent(
    name="CodeWriter",
    system_message="You are a helpful AI assistant.\n"
    "You use your coding skill to solve problems.\n"
    "You have access to a IPython kernel to execute Python code.\n"
    "You can suggest Python code in Markdown blocks, each block is a cell.\n"
    "The code blocks will be executed in the IPython kernel in the order you suggest them.\n"
    "All necessary libraries have already been installed.\n"
    "Once the task is done, returns 'TERMINATE'.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]},
)

code_executor_agent = ConversableAgent(
    name="CodeExecutor",
    llm_config=False,
    code_execution_config={"executor": NotebookExecutor()},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").strip().upper(),
)
```

Let's use the agents to complete a simple task of drawing a plot showing
the market caps of the top 7 publicly listed companies.


```python
chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message="Create a plot showing the market caps of the top 7 publicly listed companies using data from Yahoo Finance.",
)
```

    CodeExecutor (to CodeWriter):
    
    Create a plot showing the market caps of the top 7 publicly listed companies using data from Yahoo Finance.
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    To accomplish this task, we would use the `yfinance` library to fetch data from Yahoo Finance, `pandas` library for data manipulation, and `matplotlib` for data visualization.
    
    Steps:
    1. Identify the tickers for the top 7 largest publicly listed companies. Currently, these companies are: Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), Facebook (FB), Tesla (TSLA), and Berkshire Hathaway (BRK-A).
    2. Fetch market cap information for these companies using yfinance.
    3. Clean and process the fetched data into a usable format (a pandas DataFrame).
    4. Plot the market caps for these companies.
    
    Let's start by fetching data for these companies.
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeExecutor (to CodeWriter):
    
    
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    Great! Before I continue, I need to know if the necessary libraries are installed.
    
    The libraries needed for this task are:
    1. `yfinance`
    2. `pandas`
    3. `matplotlib`
    
    If these libraries are not installed, you can install them using pip:
    
    ```python
    !pip install yfinance pandas matplotlib
    ```
    
    Assuming these libraries are installed, we would import them and proceed with fetching the market cap data:
    
    ```python
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Define tickers 
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "BRK-A"]
    
    # Fetch data
    data = yf.download(tickers, start="2022-02-01", end="2022-02-28")
    
    # Extract only 'Close' values for each ticker
    data = data['Close']
    
    # Create empty dictionary to hold market cap data
    market_caps = {}
    
    # Calculate market caps
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        market_caps[ticker] = info["marketCap"]
    
    # Convert market_caps dictionary to pandas DataFrame
    df = pd.DataFrame(list(market_caps.items()), columns=['Company', 'Market_Cap'])
    
    # Sort DataFrame by Market_Cap in descending order
    df = df.sort_values('Market_Cap', ascending=False)
    
    # Plot data
    plt.figure(figsize=(10,6))
    plt.barh(df['Company'], df['Market_Cap'], color='skyblue')
    plt.xlabel('Market Cap (in trillions)')
    plt.title('Market Caps of Top 7 Publicly Listed Companies')
    plt.gca().invert_yaxis()
    plt.show()
    ```
    
    Please note that the start and end dates used while fetching data specifies the time period we are interested in. Feel free to modify these as needed. The 'marketCap' obtained from the 'info' property of the Ticker object represents the market cap as of the end date.
    
    Also note that the final plot shows the companies in descending order of market cap, with the company with the highest market cap at the top of the chart.
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    
    >>>>>>>> EXECUTING 2 CODE BLOCKS (inferred languages are [python, python])...



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[24], line 20
         18 for ticker in tickers:
         19     info = yf.Ticker(ticker).info
    ---> 20     market_caps[ticker] = info["marketCap"]
         22 # Convert market_caps dictionary to pandas DataFrame
         23 df = pd.DataFrame(list(market_caps.items()), columns=['Company', 'Market_Cap'])


    KeyError: 'marketCap'


    CodeExecutor (to CodeWriter):
    
    exitcode: 0 (execution succeeded)
    Code output: Requirement already satisfied: yfinance in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (0.2.36)
    Requirement already satisfied: pandas in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (2.2.1)
    Requirement already satisfied: matplotlib in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (3.8.3)
    Requirement already satisfied: numpy>=1.16.5 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (1.26.4)
    Requirement already satisfied: requests>=2.31 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (2.31.0)
    Requirement already satisfied: multitasking>=0.0.7 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (0.0.11)
    Requirement already satisfied: lxml>=4.9.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (5.0.1)
    Requirement already satisfied: appdirs>=1.4.4 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (1.4.4)
    Requirement already satisfied: pytz>=2022.5 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (2023.3.post1)
    Requirement already satisfied: frozendict>=2.3.4 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (2.4.0)
    Requirement already satisfied: peewee>=3.16.2 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (3.17.0)
    Requirement already satisfied: beautifulsoup4>=4.11.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (4.12.2)
    Requirement already satisfied: html5lib>=1.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from yfinance) (1.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: tzdata>=2022.7 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from pandas) (2023.4)
    Requirement already satisfied: contourpy>=1.0.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (4.47.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (23.2)
    Requirement already satisfied: pillow>=8 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (10.2.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from matplotlib) (3.1.1)
    Requirement already satisfied: soupsieve>1.2 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from beautifulsoup4>=4.11.1->yfinance) (2.5)
    Requirement already satisfied: six>=1.9 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from html5lib>=1.1->yfinance) (1.16.0)
    Requirement already satisfied: webencodings in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from html5lib>=1.1->yfinance) (0.5.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from requests>=2.31->yfinance) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from requests>=2.31->yfinance) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from requests>=2.31->yfinance) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages (from requests>=2.31->yfinance) (2024.2.2)
    /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [**************        29%%                      ]  2 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [********************* 43%%                      ]  3 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [**********************57%%*                     ]  4 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [**********************71%%********              ]  5 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [*********************100%%**********************]  7 of 7 completed
    
    1 Failed download:
    ['FB']: Exception('%ticker%: No timezone found, symbol may be delisted')
    
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    From the error message, it seems that the 'FB' ticker (Facebook) failed to download because it might have been delisted. This is likely due to Facebook's corporate rebranding to Meta Platforms Inc. in late 2021, which resulted in a ticker change from 'FB' to 'META'.
    
    To resolve this, we'll replace 'FB' in our tickers list with 'META' and then retrieve the data again. Here is the modified code:
    
    ```python
    # Define tickers 
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-A"]
    
    # Fetch data
    data = yf.download(tickers, start="2022-02-01", end="2022-02-28")
    
    # Extract only 'Close' values for each ticker
    data = data['Close']
    
    # Create empty dictionary to hold market cap data
    market_caps = {}
    
    # Calculate market caps
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        market_caps[ticker] = info["marketCap"]
    
    # Convert market_caps dictionary to pandas DataFrame
    df = pd.DataFrame(list(market_caps.items()), columns=['Company', 'Market_Cap'])
    
    # Sort DataFrame by Market_Cap in descending order
    df = df.sort_values('Market_Cap', ascending=False)
    
    # Plot data
    plt.figure(figsize=(10,6))
    plt.barh(df['Company'], df['Market_Cap'], color='skyblue')
    plt.xlabel('Market Cap (in trillions)')
    plt.title('Market Caps of Top 7 Publicly Listed Companies')
    plt.gca().invert_yaxis()
    plt.show()
    ```
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    
    >>>>>>>> EXECUTING CODE BLOCK (inferred language is python)...



    
![png](custom-executor_files/custom-executor_8_3.png)
    


    CodeExecutor (to CodeWriter):
    
    exitcode: 0 (execution succeeded)
    Code output: /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [                       0%%                      ]/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    /Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [********************* 43%%                      ]  3 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [**********************57%%*                     ]  4 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [**********************71%%********              ]  5 of 7 completed/Users/ekzhu/miniconda3/envs/autogen/lib/python3.11/site-packages/yfinance/utils.py:775: FutureWarning: The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.
      df.index += _pd.TimedeltaIndex(dst_error_hours, 'h')
    [*********************100%%**********************]  7 of 7 completed
    
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    I see that the fetched data was successfully retrieved and processed. However, it looks like the result of the plot isn't visible, so we don't know whether the plot was generated successfully. Please run the code again and provide the output of the plot.
    
    If there are any issues or any other points you would like me to help with, let me know!
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeExecutor (to CodeWriter):
    
    
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    I'm glad we were able to retrieve and process the data successfully. Please try running the last part of the code again to generate and display the plot:
    
    ```python
    # Plot data
    plt.figure(figsize=(10,6))
    plt.barh(df['Company'], df['Market_Cap'], color='skyblue')
    plt.xlabel('Market Cap (in trillions)')
    plt.title('Market Caps of Top 7 Publicly Listed Companies')
    plt.gca().invert_yaxis()
    plt.show()
    ```
    
    This section of the code creates a horizontal bar plot of the market capitalizations of the companies. The `plt.gca().invert_yaxis()` line is included to invert the y-axis, so the company with the highest market cap is at the top of the chart.
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    
    >>>>>>>> EXECUTING CODE BLOCK (inferred language is python)...



    
![png](custom-executor_files/custom-executor_8_5.png)
    


    CodeExecutor (to CodeWriter):
    
    exitcode: 0 (execution succeeded)
    Code output: 
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    I see that the code has executed successfully, but unfortunately, the generated plot is not visible here. However, given that there are no errors, it's likely that the plot has been created as expected when you executed the code on your end.
    
    If you have any other questions related to this code or need further assistance with Python coding or data visualization, please let me know! I'm here to help.
    
    Otherwise, if this completes your initial request, I will end this task. Just let me know your decision.
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeExecutor (to CodeWriter):
    
    
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> USING AUTO REPLY...
    CodeWriter (to CodeExecutor):
    
    Alright. If you have any more questions regarding this task, or if you need help with other tasks in the future, don't hesitate to ask. Have a great day!
    
    '---TERMINATE---'
    
    --------------------------------------------------------------------------------


You can see the plots are now displayed in the current notebook.
