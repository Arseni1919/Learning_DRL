from agents import Agent, Runner
import os
import openai


agent = Agent(name="Assistant", instructions=
"You are a helpful assistant")

result = Runner.run_sync(agent,
                         "Write a haiku about recursion in programming.")

print(result.final_output)

