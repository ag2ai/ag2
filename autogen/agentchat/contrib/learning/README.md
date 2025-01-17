## Intro
There is a notebook which shows this off: swarms_as_code.ipynb and swarms_as_code_empty.ipynb (use the empty one unless you want to see my logs)

The idea for this came from the fact that I wanted to leverage the swarm-loop debugging functionality
but there was a lot to be desired in the "orchestration" of these swarm-loop debugging efforts.
I'm sure someone can make a model strategize and orchestrate big complicated swarms though.

As I played with it, I wanted things like the following:
- I want to have more control about memories for agents
- I want to be able to orchestrate things in for loops
- I want it to be able to generate and run code which involves other agents and functions
- I want to be able to use these things in my code. I want it to write functions for me and I use it, in the same code
- I want it to invoke functions for me, and I get the actual result value, not just LLM output

I was trying to figure out solutions for these things, but then had the idea of removing the
outer swarm layer, and replacing it with code. Code is good! I can orchestrate some
very complicated things in code much more easily than I could with swarms. And, it makes them easier to use too.

Next are descriptions of the different features involved.

### Agents which invoke a function for you
One of the key features of swarms and group chats are their ability fix code by working through errors.
They do this by keeping the trial and error in the message history in order to debug.

The class `KnowledgeFunctionSwarmAgent` works by letting you register a function for it to be responsible for invoking.
This does mean you have to adhere to some of the AG2 function standards to give it the best chance of working well.

This code `queryer.auto_gen_func("List the available tables.")` will trigger a swarm
to run, use the function you provided, and return you a result which answers your question. (maybe should be called request)

There are a lot of features in the code like having it add a hypothesis and validation criteria
and then it validates the response against those validation criteria. The flexibility of swarms lets you define a pretty good
state machine to run the functions and validate the responses.

Right now, the agent expects that it will be able to answer the question using a single response from the function.
This is to guarantee there isn't any funky business with it hallucinating answers based on message history when debugging.

### Memory and memory sharing between agents
In order to make these useful, they need some sort of memory between invocations.

For now, everything is tied to "remembering" function calls. You must run `agent.remember(res)` in order to
have the result committed to memory. By making this step explicit, it lets the user control what exactly is remembered
by the agent at a specific point in time.

For example, you could want to do a bunch of "initialization" requests to build up memory. Help the agent get an idea of what is possible
through building it up from basic to complicated. Then, you can choose to use the agent exactly as is from then on, or remember things.

You can also register different agents like `agent.add_knowledge_source(agent_1)` to an agent, which means that `agent` will have the memories of `agent_1` on top of it's own memories.
There's some interesting stuff in the code because it's likely important for the main agent to
have the memories represented with parameters to help with future invocations, but that isn't
relevant for when those memories are used by other agents.

### Native python code generation and running

Now, what if you don't want to write your own functions?
- Sometimes I want to have an LLM just write a function for me, in my code, in the easiest way possible.
- I want it to 'just work', which means it compiles and runs for sure
- I want it the code to be written using real data from my actively running code to educate
and test the functionality.

Well it actually works, and it's actually built on top of the other agent code!

`KnowledgeCodeGenSwarmAgent` essentially defines a function for `KnowledgeFunctionSwarmAgent`
where this function is used normally by the swarm, but the function actually executes code in a jupyter kernel to test it.
The data from the pain process is pickled and passed in (gross but cool), and the function that runs in the jupyter kernel is pickled and passed
back out.

Right now you have to pass in the params, types, example value, and return type as well. But this can be improved.

```python
res = func_generator.generate_function(
            """Generate me a function which picks a random fish from the input dataframe input_df, but the choice is decided based on their size.
The bigger the fish the more common it is based on the multiplier parameter. Return (Name, Size)""",
            function_name="column_analysis_func",
            params={"input_df": (all_fish, pandas.DataFrame),
                    "multiplier": (5, int)},
            return_type=Tuple[str, str],
    )

func = res.result
func(fish, 10) # This is now a function you can use!
```

This works with memory as well, so you can educate the code generator with knowledge from other code generators or other agents.

### Future Work
Exciting code gen ideas:
- Make it functional as an annotation, so you'd just define the skeleton of the function and annotate it, and then it
becomes a function which codes itself at runtime
- Have the resulting function actually be wrapped and in a try except block, and if it throws an exception,
it will use the params that resulted in the exception and re-generate itself. I think that would be super cool.
- Figure out some way to pass in the context of the file or the class it is being invoked in. I think this begins the journey to doing more complicated gen things.

Other ideas:
- More extensive testing of the results
- The ability to get a pre-configured Callable out of the result so that you can invoke the function the same way in the future N times
- Figure out how to get it to return a "wrapped" function with different params.
For example, the input to the function is "query" for a SQL query, but I want to be able to pass in a table name and that gets inserted into the query somehow.
- Better memory management methods. Creating snapshots of memory at specific times, compaction, etc.
- Make it so that code gen gets the input params passed in as strings so it can see them
- Really cool: Instead of having code gen return a function directly, how about returning code that makes it so if the function is invoked and it throws an exception
the
