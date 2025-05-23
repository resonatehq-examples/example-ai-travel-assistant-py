from .tools import tools_config
import json

def interact_with_llm(ctx, messages):
    try:
        openai_client = ctx.get_dependency("openai_client")
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            store=False,
            messages=messages,
            tools=tools_config,
        )

        msg = completion.choices[0].message

        # Safely extract and convert tool_calls
        tool_calls = []
        if msg.tool_calls:
            for call in msg.tool_calls:
                tool_calls.append({
                    "id": call.id,
                    "type": call.type,
                    "name": call.function.name,
                    "args": json.loads(call.function.arguments)
                })

        return {
            "content": msg.content,
            "tool_calls": tool_calls or None,
        }
    except Exception as e:
        print(f"Error in interact_with_llm: {e}")
