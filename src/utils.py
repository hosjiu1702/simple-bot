from datetime import date
from agents import RunContextWrapper, Agent
from src.schema import UserQuery
from src.prompts.generic import GENERAL_INSTRUCTIONS


def generate_instructions(wrapper: RunContextWrapper[UserQuery], agent: Agent) -> str:
    return GENERAL_INSTRUCTIONS.format(query=wrapper.context.query, datetime=str(date.today()))