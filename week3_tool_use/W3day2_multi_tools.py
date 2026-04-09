import anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ============================================================
# DEFINE 3 TOOLS
# ============================================================

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
        "description": "Performs basic math calculations. Use for addition, subtraction, multiplication, division, and percentages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '500000 * 1.15' or '920000 - 310000'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_employee_info",
        "description": "Looks up employee details by name. Returns their role, department, and email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Employee's first or full name"}
            },
            "required": ["name"]
        }
    }
]

# ============================================================
# TOOL IMPLEMENTATIONS (fake data)
# ============================================================

def get_sales_data(region, quarter, year):
    fake_data = {
        "north_Q4_2025": {"revenue": 920000, "deals": 45, "avg_deal": 20444},
        "south_Q4_2025": {"revenue": 310000, "deals": 18, "avg_deal": 17222},
        "west_Q4_2025": {"revenue": 740000, "deals": 38, "avg_deal": 19473},
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
        return json.dumps({"error": f"Cannot calculate: {str(e)}"})

def get_employee_info(name):
    employees = {
        "reeshabh": {"name": "Reeshabh", "role": "Data Analyst", "department": "Analytics", "email": "reeshabh@company.com"},
        "priya": {"name": "Priya Sharma", "role": "Sales Manager", "department": "Sales", "email": "priya@company.com"},
        "amit": {"name": "Amit Patel", "role": "Engineering Lead", "department": "Engineering", "email": "amit@company.com"},
    }
    key = name.lower().split()[0]
    if key in employees:
        return json.dumps(employees[key])
    return json.dumps({"error": f"Employee '{name}' not found"})

# Map tool names to functions
tool_functions = {
    "get_sales_data": get_sales_data,
    "calculator": calculator,
    "get_employee_info": get_employee_info,
}

# ============================================================
# FUNCTION TO HANDLE THE FULL TOOL USE FLOW
# ============================================================

def ask_with_tools(question):
    print(f"\nUser: {question}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        tools=tools,
        messages=[{"role": "user", "content": question}]
    )

    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")

        print(f"  → Claude chose: {tool_block.name}({json.dumps(tool_block.input)})")

        # Run the actual function
        func = tool_functions[tool_block.name]
        result = func(**tool_block.input)
        print(f"  → Tool returned: {result}")

        # Send result back to Claude
        final = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0,
            tools=tools,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tool_block.id, "content": result}
                ]}
            ]
        )
        print(f"  → Claude: {final.content[0].text}")
    else:
        print(f"  → Claude: {response.content[0].text}")

# ============================================================
# TEST WITH DIFFERENT QUESTIONS
# ============================================================

print("=" * 60)
print("  MULTI-TOOL TEST: Claude picks the right tool")
print("=" * 60)

ask_with_tools("What were the sales numbers for North in Q4 2025?")
ask_with_tools("What is 920000 minus 310000?")
ask_with_tools("Who is Priya and what's her email?")
ask_with_tools("If North had $920K revenue and grew 15%, what would next quarter look like?")
ask_with_tools("What's the weather in Delhi?")