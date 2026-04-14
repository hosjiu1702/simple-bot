from dataclasses import dataclass
import asyncio
import logfire
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI
from agents import (
    Agent,
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
from dotenv import load_dotenv
import os
import textwrap
from typing import List, Optional
from datetime import date


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
openai_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
litellm_client = LitellmModel(MODEL_NAME, os.getenv("LITELLM_BASE_URL"), os.getenv("LITELLM_API_KEY"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

anthropic_client = Anthropic(base_url=ANTHROPIC_GENERIC_URL, api_key=ANTHROPIC_API_KEY)
set_tracing_disabled(disabled=True)


###### LLM OBSERVABILITY ######
logfire.configure()
logfire.instrument_openai_agents()

if DEBUG:
    from pprint import pprint
    pprint(f"\n[ANTHROPIC CLIENT CONFIGURATION]\n{vars(anthropic_client)}")


@dataclass
class LatestUserMessage:
    message: str

@dataclass
class ImageQuery:
    url: str
    query: str

########### TOOL DEFINITION ##########
# Tool definition using OpenAI SDK which under the hood
# using Anthropic client to directly call the built-in web_search tool.
@function_tool
def search_web(query: str):
    print("Called `search_web` tool.")
    print(f"query: {query}")
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
    # output = dict({
    #     "content": [{
    #         "type": "text",
    #         "text": f"{response.content[0].text}"
    #         }
    #     ]
    # })
    return output


@function_tool
def analyze_image(wrapper: RunContextWrapper[ImageQuery]) -> str:
    """ Analyze the image content for a given user query.
    """
    image_url = wrapper.context.url
    image_query = wrapper.context.query
    # print(f"[DEBUG] Latest User Message: {wrapper.context.message}")
    print(f"[DEBUG][TOOL] Called analyze_image.")
    print(f"[DEBUG][TOOL] Image URL: {image_url}")
    print(f"[DEBUG][TOOL] Image Query: {image_query}")

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

@function_tool
def generate_image(prompt: str):
    return "here is the generated image: 😁"


########## PROMPT ##########
# INSTRUCTIONS = textwrap.dedent(f"""
# I'm Luky 🐶 which are a helpful assistant that can search the internet given the user query.

# - Your answer must be neat and concise.
# - Call search_web tool when needed.
# - Always use vietnamese for the final answer.

# Curernt datetime: {str(date.today())}
# """).strip()

INSTRUCTIONS = textwrap.dedent(f"""
I'm Luky 🐶 which are a helpful assistant that can search the internet or analyse an image given the user query.

- Your answer must be neat and concise.
- Call search_web tool when needed.
- When user asking or discussing about any image, call analyze_image tool.
- If input query is *IGNORE*. Return only exact *Nothing*.
- Always use vietnamese for the final answer.

Curernt datetime: {str(date.today())}
""").strip()


NEWS_AGENT_INSTRUCTIONS = INSTRUCTIONS


@dataclass
class LLMClient(str):
    OpenAI = "openai"
    LiteLLM = "litellm"


class NewsAgent:
    def __init__(
        self,
        agent_name: str = "News Agent",
        client: AsyncOpenAI | LitellmModel | str = client,
        model_name: str = MODEL_NAME,
        instructions: str = NEWS_AGENT_INSTRUCTIONS,
        tools: List[FunctionTool] = [search_web, generate_image, analyze_image],
        debug: bool = False,
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

        self.agent = Agent(
            name=agent_name,
            instructions=instructions,
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
        photo_url: Optional[str] = None
    ):
        # get conversation history in this session
        # and then return the latest user message.
        # items = await session.get_items(limit=1)
        # if len(items):
        #     latest_msg = items[-1]
        # else:
        #     latest_msg = ""
        # print(f"[DEBUG][agent.py] latest_msg: {latest_msg}")
        # user send photo without any caption
        print(f"[DEBUG][agent.py] query: {query}")

        context = None
        if photo_url:
            context = ImageQuery(url=photo_url, query=query)

        result = await Runner.run(
            self.agent,
            query,
            session=session,
            context=context
        )

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
