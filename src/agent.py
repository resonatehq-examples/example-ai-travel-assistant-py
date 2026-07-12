import asyncio
import json
import os
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI
from resonate.resonate import Resonate

from .llm import interact_with_llm
from .tools import (
    scrape_website,
    search_internet,
)

if TYPE_CHECKING:
    from resonate.context import Context

load_dotenv()


async def travel_assistent(ctx: "Context"):
    messages = [
        {
            "role": "system",
            "content": """
                You are a travel assistant.
                You are capable of the full end-to-end travel planning process.
                You can search the internet, scrape websites, and interact with the user, whose trip you are planning.
                You should be very detailed in any itenerary, dates, costs, and others suggestions you provide, including where to stay, what flights to take, and what to do.
                You should try to align the trip with the user's interests and preferences as much as possible.
                Make sure to ask for dates, locations, and what they like to do.
                But you should also be very creative and suggest things that the user might not have thought of.
                If you are not sure about something, ask the user for more information.
                If you need to search the internet for information, do so.
                If you need to scrape a website for information, do so.

                When you have a complete trip plan, submit say "TRIP PLANNING COMPLETE" and then say the full plan.
                Only say "TRIP PLANNING COMPLETE" when you are sure that the trip is complete, and have no more questions for the user.
            """,
        },
        {
            "role": "user",
            "content": """
                 Plan a trip for me.
            """,
        },
    ]

    while True:
        message = await ctx.run(interact_with_llm, messages)
        # Always add the assistant response
        assistant_message = {"role": "assistant", "content": message["content"]}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": call["type"],
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    },
                }
                for call in message["tool_calls"]
            ]
        messages.append(assistant_message)
        content = message.get("content")
        if content and "TRIP PLANNING COMPLETE" in content:
            break
        elif message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                if tool_name == "internet_search":
                    result = await ctx.run(
                        search_internet,
                        args["search_query"],
                        args.get("num_results", 5),
                    )
                elif tool_name == "scrape_website":
                    result = await ctx.run(scrape_website, args["url"])
                else:
                    result = "Unknown tool call"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )
        elif content:
            input_message = await ctx.run(chat_with_user, content)
            messages.append({"role": "user", "content": input_message})

    return message["content"]


def chat_with_user(_, text_from_llm: str) -> str:
    return input(dedent(text_from_llm))


async def _main() -> None:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    serper_api_key = os.environ.get("SERPER_API_KEY")
    browserless_api_key = os.environ.get("BROWSERLESS_API_KEY")

    if not serper_api_key:
        sys.exit("SERPER_API_KEY environment variable is required")
    if not browserless_api_key:
        sys.exit("BROWSERLESS_API_KEY environment variable is required")

    openai_client = OpenAI(api_key=openai_api_key)

    resonate = Resonate(
        url=os.environ.get("RESONATE_URL", "http://localhost:8001"),
        group="worker",
    )
    resonate.with_dependency(openai_client)
    resonate.register(travel_assistent)
    resonate.register(interact_with_llm)
    resonate.register(search_internet)
    resonate.register(scrape_website)
    resonate.register(chat_with_user)

    trip_id = input(dedent("Enter a trip Name/ID: "))
    handle = resonate.run(trip_id, travel_assistent)
    result = await handle.result()
    print("Here is a plan for your trip:")
    print(result)
    await resonate.stop()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
