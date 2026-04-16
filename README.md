# simple-agent
Like its name, this's an simple QA Zalo Agent serving for my parents.

*[WIP]*

## Installation & Usage
-  I am working on it (*maybe*). But I think you should give a quick look to [python-zalo-bot](https://pypi.org/project/python-zalo-bot/) SDK first :) 

### Technical Overview
- **OpenAI Python SDK**.
- **LiteLLM** for LLM model providers routing & cost tracking.
- **Zalo API** for interating with Zalo Bot.
- *Webhook* vs *Long-Polling* for communication protocol.
- Short-term file-based memory with SQLite database.
- Currently, I am using **GLM**, **Gemini** and **Claude** as LLM models under the hood.

### Tools
- Buil-in web search tool.

### Agent Patterns
- Single Agent currently.

### AI Tools
- **Cursor** (chat w/ the codebase directly)

### Learning List
- Async in Python.