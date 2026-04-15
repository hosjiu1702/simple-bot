import os
from dotenv import load_dotenv
from zai import ZaiClient
import requests

load_dotenv()
# response = requests.post(
#     url=os.getenv("GLM_BASE_URL") + "/images/generations",
#     headers={
#         "Content-Type": "application/json"
#     },
#     json={
#         "model": "glm-image",
#         "prompt": "A cute little kitten sitting on a sunny windowsill, with the background of blue sky and white clouds.",
#         "size": "512x512"
#     }
# )

# print(response.content)

response = requests.post(
    url=os.getenv("GLM_BASE_URL") + "/web_search",
    headers={"Content-Type": "application/json"},
    json={
        "search_engine": "search-prime",
        "search_query": "hot news today",
        "count": 5,
        "search_recency_filter": "noLimit",
    }
)

print(response.content)