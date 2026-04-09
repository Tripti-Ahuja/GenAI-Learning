import anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

tools = [
    {
        "name": "get_sales_data",
        "description": "Retrieves sales revenue and deal count for a specific region and quarter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Sales region: north, south, east, west"},
                "quarter": {"type": "string", "description": "Quarter: Q1, Q2, Q3, Q4"},
                "year": {"type": "integer", "description": "Year as 4-digit number"}
            },
            "required": ["region", "quarter", "year"]
        }
    },
    {
        "name": "calculator",
        "description": "Performs math calculations. Use for addition, subtraction, multiplication, division, percentages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '920000 * 0.15' or '920000 - 310000'"}
            },
            "required": ["expression"]
        }
    }
]

def get_sales_data(region, quarter, year):
    fake_data = {
        "north_Q4_2025": {"revenue": 920000, "deals": 45, "avg_deal": 20444},
        "south_Q4_2025": {"revenue": 310000, "deals": 18, "avg_deal": 17222},
        "north_Q3_2025": {"revenue": 800000, "deals": 40, "avg_deal": 20000},
        "south_Q3_2025": {"revenue": 350000, "deals": 22, "avg_deal": 15909},
    }
    key = f"{region.lower()}_{quarter}_{year}"
    if key in fake_data:
        return json.dumps({"region": region, "quarter": quarter, "year": year, **fake_data[key]})
    return json.dumps({"error": f"No data found for {region} {quarter} {year}"})

def calculator(expression):
    try:
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

tool_functions = {
    "get_sales_data": get_sales_data,
    "calculator": calculator,
}

# ============================================================
# TOOL CHAINING LOOP — keeps going until Claude is done
# ============================================================

def ask_with_chaining(question):
    print(f"\nUser: {question}")
    print("-" * 50)

    messages = [{"role": "user", "content": question}]
    step = 1

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0,
            tools=tools,
            messages=messages
        )

        # If Claude is done (no more tools to call), print final answer
        if response.stop_reason == "end_turn":
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\nFinal Answer: {final_text}")
            break

        # Process all tool calls in this response
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = tool_functions[block.name]
                result = func(**block.input)
                print(f"  Step {step}: {block.name}({json.dumps(block.input)}) → {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
                step += 1

        # Add Claude's response and tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

# ============================================================
# TEST QUESTIONS THAT NEED MULTIPLE TOOLS
# ============================================================

print("=" * 60)
print("  TOOL CHAINING: Multi-step problem solving")
print("=" * 60)

# Needs: get_sales_data → calculator
ask_with_chaining("What is the difference in revenue between North and South in Q4 2025?")

# Needs: get_sales_data → get_sales_data → calculator
ask_with_chaining("How much did North region's revenue grow from Q3 to Q4 2025? Show the percentage.")

# Needs: get_sales_data → calculator
ask_with_chaining("If South Q4 2025 revenue grows by 25% next quarter, what would it be?")