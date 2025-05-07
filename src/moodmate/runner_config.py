from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from agents.run import RunConfig
from agent_hello.secrets import secrets
from rich import print


external_client = AsyncOpenAI(
    api_key=secrets["GEMINI_API_KEY"],
    base_url=secrets["GEMINI_API_URL"],
)

model = OpenAIChatCompletionsModel(
    model=secrets["GEMINI_API_MODEL"],
    openai_client=external_client,
)


def main():
    agent = Agent(
        name="Assistant",
        instructions="A helpful assistant that can answer questions and provide information.",
    )

    config = RunConfig(
        model=model,
        tracing_disabled=True,
    )

    result = Runner.run_sync(agent, "What is capital of Pakistan?", run_config=config)

    print("\n")
    print(result.final_output)
