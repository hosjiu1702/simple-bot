from dataclasses import dataclass
from typing import Any


@dataclass
class UserQuery:
    query: str
    url: str = ""
    chat_id: str = ""
    zalo_bot: Any = None