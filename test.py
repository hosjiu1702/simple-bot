import anthropic
import pprint
from dotenv import load_dotenv


load_dotenv()
DEBUG = True

client = anthropic.Anthropic(
    base_url="https://api.anthropic.com"
)

if DEBUG:
    pprint.pprint(vars(client))

query = "quickly summarize about the liteLLM's supply chain attack recently."
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": query,
        }
    ],
    tools=[{"type": "web_search_20260209", "name": "web_search"}],
)

print(response.content)