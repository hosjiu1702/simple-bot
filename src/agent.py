from dataclasses import dataclass
import asyncio
import logfire
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI
from agents import (
    Agent,
    handoff,
    Handoff,
    FunctionTool,
    ModelSettings,
    SQLiteSession,
    RunContextWrapper,
    WebSearchTool,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool
)
from agents.extensions.models.litellm_model import LitellmModel
import litellm
import requests
from zai import ZaiClient
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import textwrap
from typing import List, Optional, Annotated
from datetime import date
from src.prompts.generic import GENERAL_INSTRUCTIONS
from src.schema import UserQuery
from src.utils import generate_instructions
from phone_agent import IOSPhoneAgent
from phone_agent.model import ModelConfig
from phone_agent.agent_ios import IOSAgentConfig


# Local debug session.
DEBUG = False


######## SET UP ENVIRONMENT VARIABLES ########
load_dotenv()

# Connect to OpenAI / Anthropic server directly?
# BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
# API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BASE_URL = os.getenv("LITELLM_BASE_URL")
API_KEY = os.getenv("LITELLM_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

ANTHROPIC_GENERIC_URL = os.getenv("ANTHROPIC_GENERIC_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
# LiteLLM Proxy Server

# LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")
# LITELLM_API_KEY = os.getenv("LOCAL_ANTHROPIC_API_KEY", "")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set value for either ANTHROPIC_BASE_URL or ANTHROPIC_API_KEY or MODEL_NAME via .env")

# ----- IGNORE ------
# Because Anthropic has a OpenAI-compatibility layer
# So we still be able to use OpenAI Python SDK to communicate
# with Anthropic models that have some certain limitations.
#
# In addition, we use LiteLLM as a intermediate proxy server
# between the OpenAI client and the OpenAI server.
# ----- IGNORE -----

####### OpenAI-compatible client initialization #######
# openai_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
litellm_client = LitellmModel(MODEL_NAME, os.getenv("LITELLM_BASE_URL"), os.getenv("LITELLM_API_KEY"))
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
zai_client = ZaiClient(base_url=os.getenv("GLM_BASE_URL"), api_key="")
anthropic_client = Anthropic(base_url=ANTHROPIC_GENERIC_URL, api_key=ANTHROPIC_API_KEY)
set_tracing_disabled(disabled=True)


###### LLM OBSERVABILITY ######
# logfire.configure()
# logfire.instrument_openai_agents()

if DEBUG:
    from pprint import pprint
    pprint(f"\n[ANTHROPIC CLIENT CONFIGURATION]\n{vars(anthropic_client)}")


@dataclass
class LatestUserMessage:
    message: str


########### TOOL DEFINITION ##########
# Tool definition using OpenAI SDK which under the hood
# using Anthropic client to directly call the built-in web_search tool.
@function_tool
def search_web(query: str):
    print("Called `search_web` tool.")
    print(f"query: {query}")

    if "claude" in MODEL_NAME:
        # we use anthropic sdk here
        # and call to the built-in web search tool.
        response = anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[{"role": "user", "content": query}],
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            output_config={"effort": "low"},
        )

        if DEBUG:
            print(f"\n[WEB SEARCH TOOL OUTPUT]\n{response.content}")

        # print(f"\n#\n#\n#\n: {response}\n\n{response.content[-1].text}")
        # https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/programmatic_tool_calling_ptc.ipynb
        output = "".join([block.text for block in response.content if isinstance(block, TextBlock)])
        print(f"\n#\n#\n#\n: {response}\n\n{output}")
        return output
    
    if "glm" in MODEL_NAME:
        print(f"[DEBUG][agent.py][search_web] GLM called.")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": query}],
            tools=[{
                "type": "web_search",
                "web_search": {
                    "enable": "True",
                    "search_engine": "search-prime",
                    "search_result": "True",
                    "search_prompt": "You are a helful searcher.",
                    "count": "5",
                    "search_recency_filter": "noLimit"
                }
            }]
        )
        return ""

    if "gemini" in MODEL_NAME:
        if "openrouter" in MODEL_NAME:
            # Because Google doesn't support for Openrouter now
            # so we need to work around here by calling directly
            # to Openrouter server to use web search feature
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv("OPENROUTER_API_KEY")}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    "tools": [
                        {"type": "openrouter:web_search"},
                        {"type": "openrouter:datetime"}
                    ]
                }
            )
            data = response.json()
            text_output = data["choices"][0]["message"]["content"]
            return text_output
        else:
            response = requests.post(
                url=f"{os.getenv("LITELLM_BASE_URL")}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv("LITELLM_API_KEY")}"
                },
                json={
                    "model": f"gemini-2.5-flash",
                    "messages": [{"role": "user", "content": query}],
                    "tools": [{"googleSearch": {}}]
                }
            )
            text_output = response.json()["choices"][0]["message"]["content"]
            print(f"[DEBUG][agent.py][search_web] GEMINI CALLED.\n[Content]: {text_output}")
            return text_output

    return ""


@function_tool
def analyze_image(wrapper: RunContextWrapper[UserQuery]) -> str:
    """ Analyze the image content for a given user query.
    """
    image_url = wrapper.context.url
    image_query = wrapper.context.query
    # print(f"[DEBUG] Latest User Message: {wrapper.context.message}")
    print(f"[DEBUG][TOOL] Called analyze_image.")
    print(f"[DEBUG][TOOL] Image URL: {image_url}")
    print(f"[DEBUG][TOOL] Image Query: {image_query}")

    if "claude" in MODEL_NAME:
        # Call LLM to analyze the image
        # Anthropic-specific config input
        response = anthropic_client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_url
                            }
                        },
                        {"type": "text", "text": image_query}
                    ]
                }
            ]
        )
        output = "".join([block.text for block in response.content if isinstance(block, TextBlock)])
        print(f"[DEBUG][TOOL][anylyze_image]\n#\n#\n#\n: {response}\n\n{output}")
        return output

    if "gemini" in MODEL_NAME:
        pass


@function_tool
def generate_image(prompt: str):
    return "here is the generated image: 😁"


@function_tool
def book_a_ride(
    pickup: Annotated[str, "the pickup location"],
    destination: Annotated[str, "the dropoff location"]
):
    """
    Book a ride given the pickup and destination locations.
    """
    print(f"[DEBUG][book_a_ride] Called.")
    print(f"[DEBUG][book_a_ride] pickup: {pickup}")
    print(f"[DEBUG][book_a_ride] destination: {destination}")

    model_config = ModelConfig(
        base_url=os.getenv("PHONE_AGENT_BASE_URL"),
        api_key=os.getenv("PHONE_API_KEY"),
        model_name=os.getenv("PHONE_MODEL_NAME")
    )
    agent_config = IOSAgentConfig(
        wda_url=os.getenv("PHONE_AGENT_WDA_URL"),
        lang="en",
        verbose=True
    )
    phone_agent = IOSPhoneAgent(
        model_config=model_config,
        agent_config=agent_config
    )

    BOOKING_QUERY = f"""
    open Grab app, and book a ride (vehicle type is only bike) in which 
    the current location is {pickup} and the target is {destination}.
    """
    print(f"[DEBUG][book_a_ride] running ...")
    result = phone_agent.run(BOOKING_QUERY)

    return result


@dataclass
class LLMClient(str):
    OpenAI = "openai"
    LiteLLM = "litellm"


# Create a dedicated phone agent for phone using and
# take over the control of the conversation flow.
ride_booking_agent = Agent(
    name="Ride Booking Agent",
    instructions="You are a helpful assistant which help to book a ride for user. Always call book_a_ride tool.",
    handoff_description="Whenever user asks for ride booking then this agent will be invoked and take over the conversation control.",
    tools=[book_a_ride],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    model_settings=ModelSettings(max_tokens=1024),
)

class JourneyInfo(BaseModel):
    pickup: str
    dropoff: str

async def ride_booking_handoff(wrapper: RunContextWrapper[UserQuery], input_data: JourneyInfo):
    """ Notify back to the user that the booking process just get started """
    ctx = wrapper.context
    pickup = input_data.pickup
    dropoff = input_data.dropoff
    msg = f"From (điểm đi): {pickup}\n"
    msg += f"To (điểm đến): {dropoff}"
    await ctx.zalo_bot.send_message(ctx.chat_id, msg)
    await asyncio.sleep(1.5)
    await ctx.zalo_bot.send_message(ctx.chat_id, "⏳")

ride_booking_handoff = handoff(
    agent=ride_booking_agent,
    on_handoff=ride_booking_handoff,
    input_type=JourneyInfo
)

class NewsAgent:
    def __init__(
        self,
        agent_name: str = "News Agent",
        client: AsyncOpenAI | LitellmModel | str = client,
        model_name: str = MODEL_NAME,
        instructions: str = GENERAL_INSTRUCTIONS,
        tools: List[FunctionTool] = [search_web, generate_image, analyze_image],
        debug: bool = False,
        handoffs: Optional[List] = [ride_booking_handoff]
    ):
        # if isinstance(client, str):
        #     if client == LLMClient.OpenAI:
        #         self.client = openai_client
        #     elif client == LLMClient.LiteLLM:
        #         self.client = litellm_client
        #     else:
        #         raise ValueError(f"We do not support value {client}. Supported ones are {LLMClient.OpenAI}, {LLMClient.LiteLLM}.")
        # elif isinstance(client, (AsyncOpenAI, LitellmModel)):
        #     self.client = client
        # else:
        #     raise ValueError(f"We do not support value {client}. Supported ones are {LLMClient.OpenAI}, {LLMClient.LiteLLM}.")

        self.agent = Agent[UserQuery](
            name=agent_name,
            instructions=generate_instructions,
            handoffs=handoffs,
            tools=tools,
            model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
            # model=LitellmModel(MODEL_NAME, "https://api.anthropic.com", API_KEY), # LiteLLM as client
            model_settings=ModelSettings(
                max_tokens=1024,
                extra_body={
                    "reasoning_effort": "low" # only for anthropic models
                }
            )
        )
        if debug:
            import litellm
            litellm._turn_on_debug()
            print(f"[DEBUG][agent.py] Initialized agent.")

    async def reply(
        self,
        query: str,
        session: SQLiteSession,
        photo_url: Optional[str] = None,
        chat_id: Optional[str] = "",
        zalo_bot: Optional = None
    ):
        print(f"[DEBUG][agent.py] query: {query}")

        # Add context for two purposes:
        #   1. Getting the Image URL
        #   2. Getting the query for detecting language input
        context = UserQuery(url=photo_url, query=query) if photo_url else UserQuery(query=query, chat_id=chat_id, zalo_bot=zalo_bot)
        result = await Runner.run(self.agent, query, context=context, session=session)

        print(f"[DEBUG][agent.py] final_output: {result.final_output}")
        return result.final_output



async def main():
    news_agent = Agent(
        name="News Agent",
        instructions=INSTRUCTIONS,
        tools=[search_web],
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)
    )
    query = "thông tin về cuộc tấn công mã độc nhắm vào thư viện LiteLLM gần đây."
    result = await Runner.run(news_agent, query)

    if DEBUG:
        from pprint import pprint
        pprint(vars(result))

    print(result.final_output)


if __name__ == "__main__":
    # asyncio.run(main())
    news_agent = NewsAgent()
    result = asyncio.run(news_agent.reply("Latest news about Elon Musk today."))
    print(result)
