import asyncio
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from agents.run import RunConfig
from moodmate.secrets import secrets
from rich import print


external_client = AsyncOpenAI(
    api_key=secrets["GEMINI_API_KEY"],
    base_url=secrets["GEMINI_API_URL"],
)

model = OpenAIChatCompletionsModel(
    model=secrets["GEMINI_API_MODEL"],
    openai_client=external_client,
)


async def main():
    agent = Agent(
        name="MoodMate",
        instructions="""
    You are MoodMate, a friendly and emotionally intelligent assistant.
    Your job is to help users reflect on their emotions and offer general tips for self-care.
    Be supportive, non-judgmental, and never offer medical advice.
    Use soft, comforting language and speak like a calm friend or life coach.
    Always end your response with a gentle question to encourage continued reflection.
    """,
    )

    config = RunConfig(
        model=model,
        tracing_disabled=True,
    )

    result = await Runner.run(agent, "I've been feeling anxious and low energy lately.", run_config=config)

    print("\n")
    print(result.final_output)
    with open("output.md", "w", encoding="utf-8") as f:
        f.write(result.final_output)

def run_main():
    asyncio.run(main())