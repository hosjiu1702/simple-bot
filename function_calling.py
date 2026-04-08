# how can we train or finetune for a simple reasoning model
# that can solve your specific problems
# ---------
# 
# This uses OpenAI SDK to build a simple LLM-based assistant.
from openai import OpenAI
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from pprint import pprint
import json


@dataclass
class BookingStatus:
    status: str = "pending"


# Tools Definition
def book_a_ride(pickup_location: str, dropoff_location: str):
    """ Book a ride for a given pickup and dropoff location, which
    is provided from the user query.

    Args:
        pickup_location: current user location.
        dropoff_location: destination of the ride.
    
    Returns:
        booking_status: status of ride booking.
    """
    # Just for testing.
    print(f"pickup_location: {pickup_location}")
    print(f"dropoff_location: {dropoff_location}")
    return f"Here is your booking status: SUCCESS"


def get_fn_by_name(fn_name: str):
    if fn_name == "book_a_ride":
        return book_a_ride


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_a_ride",
            "description": "Book a ride that given pickup and dropoff location user provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pickup_location": {
                        "type": "string",
                        "description": "Current location where the user is."
                    },
                    "dropoff_location": {
                        "type": "string",
                        "description": "The destination which the user would like to go to."
                    }
                },
                "required": ["pickup_location", "dropoff_location"]
            }
        }
    }
]


if __name__ == "__main__":
    # Load environment variables for security
    load_dotenv()

    MODEL_URL = os.getenv("ANTHROPIC_BASE_URL", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "")
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    print(f"MODEL NAME: {MODEL_NAME}")
    print(f"MODEL URL: {MODEL_URL}")
    assert MODEL_URL != "", "Please recheck MODEL URL in your environment variables .env"

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=MODEL_URL
    )
    print(f"\nEstablished connection to {MODEL_URL}.")

    query = "who is the president of American in 2020?"
    query2 = "tôi muốn đặt xe đi từ 42 lê công kiều đến 65 hải phòng."
    messages = [{"role": "user", "content": query2}]
    # Model Inference
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    pprint(response.choices[0])
    pprint("\n")
    pprint(f"Response Output: {response.choices[0].message.model_dump()}")

    # Add tool calls output (if any) back to messages list
    messages.append(response.choices[0].message.model_dump())

    # tools output parsing
    if tool_calls := response.choices[0].message.tool_calls:
        for tool_call in tool_calls:
            if tool_call.type != "function":
                continue

            # get function name and its arguments from LLM output
            # then execute it.
            # WARNNING: the content from LLM returns is always string.
            tool_call_id = tool_call.id
            fn_name: str = tool_call.function.name
            fn_arguments: dict = json.loads(tool_call.function.arguments)
            fn_output = json.dumps(get_fn_by_name(fn_name)(**fn_arguments))
            print(f"\nfn_output: {fn_output}")

            # update messages list for later use with the LLM
            messages.append({
                "role": "tool",
                "content": fn_output,
                "tool_call_id": tool_call_id
            })
            pprint(messages)

        # send everything back to the LLM for final answer generation.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS
        )

    print(f'\n{response}')
    print(f"\nFINAL RESPONSE: {response.choices[0].message.content}")