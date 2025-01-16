# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
from datetime import datetime

from autogen import ReasoningAgent, ThinkNode, UserProxyAgent


def load_data(file_name):
    with open(file_name, "r") as f:
        return json.load(f)


def run_benchmark(eval_list):
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "Please set the OPENAI_API_KEY environment variable."

    config_list = [{"model": "gpt-4o-mini", "api_key": api_key}]

    reasoning_agent = ReasoningAgent(
        name="reasoning_agent",
        llm_config={"config_list": config_list},
        reason_config={"method": "lats", "nsim": 2, "max_depth": 3},
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    results = []

    for i, eval_set in enumerate(eval_list):
        eval_question = eval_set["prompt"]
        question_id = eval_set["question_id"]
        ground_truth = eval_set["answer"]
        prompt = """Give the final answer using the following format: `Final Answer: X` where X is one of the letters A, B, C, D, E, or F."""

        question = eval_question + prompt

        ans = user_proxy.initiate_chat(reasoning_agent, message=question, summary_method=last_meaningful_msg)
        summary = ans.summary
        match = re.search(r"Final Answer: ([A-F])", summary)

        if match:
            extracted_answer = match.group(1)
            results.append(
                {
                    "question_id": question_id,
                    "answer": ground_truth,
                    "generated_answer": extracted_answer,
                    "summary": summary,
                }
            )
        else:
            print("No Final Answer found in the response.")
            results.append(
                {
                    "question_id": question_id,
                    "answer": ground_truth,
                    "generated_answer": "No Final Answer found in the response.",
                    "summary": summary,
                }
            )

        # uncomment to save reasoning tree
        # data = reasoning_agent._root.to_dict()
        # with open(f"reasoning_tree_{i}.json", "w") as f:
        #     json.dump(data, f)

    with open(f"results-{reasoning_agent.method}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}.json", "w") as f:
        json.dump(results, f, indent=4)


def last_meaningful_msg(sender, recipient, summary_args):
    import warnings

    if sender == recipient:
        return "TERMINATE"

    summary = ""
    chat_messages = recipient.chat_messages[sender]

    for msg in reversed(chat_messages):
        try:
            content = msg["content"]
            if isinstance(content, str):
                summary = content.replace("TERMINATE", "")
            elif isinstance(content, list):
                # Remove the `TERMINATE` word in the content list.
                summary = "\n".join(
                    x["text"].replace("TERMINATE", "") for x in content if isinstance(x, dict) and "text" in x
                )
            if summary.strip().rstrip():
                return summary
        except (IndexError, AttributeError) as e:
            warnings.warn(f"Cannot extract summary using last_msg: {e}. Using an empty str as summary.", UserWarning)
    return summary


if __name__ == "__main__":
    data = load_data("simplebench_public.json")
    run_benchmark(data["eval_data"])
