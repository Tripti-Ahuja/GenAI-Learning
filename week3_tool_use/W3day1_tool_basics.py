import anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# STEP 1: Define your tools (what Claude CAN call)
# ============================================================

tools = [
    {
        "name": "get_sales_data",
        "description": "Retrieves sales data for a given region and time period from the company database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Sales region: north, south, east, or west"
                },
                "quarter": {
                    "type": "string",
                    "description": "Quarter in format Q1, Q2, Q3, or Q4"
                },
                "year": {
                    "type": "integer",
                    "description": "Year as a 4-digit number"
                }
            },
            "required": ["region", "quarter", "year"]
        }
    }
]

# ============================================================
# STEP 2: Simulate the tool (fake data for now)
# ============================================================

def get_sales_data(region, quarter, year):
    fake_data = {
        "north_Q4_2025": {"revenue": 920000, "deals": 45, "avg_deal": 20444},
        "south_Q4_2025": {"revenue": 310000, "deals": 18, "avg_deal": 17222},
        "east_Q4_2025": {"revenue": 580000, "deals": 32, "avg_deal": 18125},
        "west_Q4_2025": {"revenue": 740000, "deals": 38, "avg_deal": 19473},
    }
    key = f"{region.lower()}_{quarter}_{year}"
    if key in fake_data:
        data = fake_data[key]
        return json.dumps({"region": region, "quarter": quarter, "year": year, **data})
    return json.dumps({"error": f"No data found for {region} {quarter} {year}"})

# ============================================================
# STEP 3: Send a question and let Claude decide to use the tool
# ============================================================

user_question = "What were the sales numbers for the North region in Q4 2025?"

print(f"User: {user_question}\n")

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    temperature=0,
    tools=tools,
    messages=[{"role": "user", "content": user_question}]
)

# ============================================================
# STEP 4: Check if Claude wants to use a tool
# ============================================================

print(f"Stop Reason: {response.stop_reason}")

if response.stop_reason == "tool_use":
    # Claude wants to call a tool
    tool_block = next(block for block in response.content if block.type == "tool_use")

    print(f"Claude wants to call: {tool_block.name}")
    print(f"With arguments: {json.dumps(tool_block.input, indent=2)}")

    # STEP 5: YOUR CODE runs the actual function
    result = get_sales_data(**tool_block.input)
    print(f"\nTool returned: {result}")

    # STEP 6: Send the result back to Claude for the final answer
    final_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        tools=tools,
        messages=[
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result
                }
            ]}
        ]
    )

    print(f"\nClaude: {final_response.content[0].text}")

else:
    # Claude answered directly without using a tool
    print(f"Claude: {response.content[0].text}")