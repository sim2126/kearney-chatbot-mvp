import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from io import StringIO
import sys
import json

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the backend/.env file.")
genai.configure(api_key=api_key)

DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Sugar_Spend_Data.csv')
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found. Make sure 'Sugar_Spend_Data.csv' is in 'backend/app/data/'")

buffer = StringIO()
df.info(buf=buffer)
df_schema = buffer.getvalue()

SYSTEM_INSTRUCTION = """
You are a Python code generator for data analysis. Generate ONLY executable Python code, nothing else.

RULES:
1. Output ONLY Python code - no markdown, no code blocks, no explanations
2. Do NOT use f-strings - use str() and concatenation with +
3. Do NOT import anything - json, pd, and df are already available
4. End with exactly ONE print() statement that outputs JSON
5. The JSON must have: {"answer": "...", "chart": null or {...}}

EXAMPLES:

User asks: "Plot the spend for each commodity"
YOUR CODE:
spend = df.groupby('Commodity')['Spend (USD)'].sum()
chart_data = {"type": "bar", "labels": spend.index.tolist(), "data": spend.values.tolist()}
result = {"answer": "Here is the spend by commodity.", "chart": chart_data}
print(json.dumps(result))

User asks: "List the commodities"
YOUR CODE:
commodities = df['Commodity'].unique().tolist()
answer = "The commodities are: " + ", ".join(commodities)
result = {"answer": answer, "chart": None}
print(json.dumps(result))

User asks: "What is the total spend?"
YOUR CODE:
total = df['Spend (USD)'].sum()
answer = "The total spend is $" + str(round(total, 2))
result = {"answer": answer, "chart": None}
print(json.dumps(result))

CHART TYPES:
- bar: {"type": "bar", "labels": [...], "data": [...]}
- pie: {"type": "pie", "labels": [...], "data": [...]}

Generate code NOW:
"""

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=SYSTEM_INSTRUCTION
)

USER_PROMPT_TEMPLATE = """
DataFrame 'df' is available with columns shown below.
pandas is imported as 'pd'.
json module is already imported and available.

DO NOT IMPORT ANYTHING - all modules are pre-loaded.

Schema:
{schema}

Chat History:
{chat_history}

User Question: {question}

Generate Python code (no imports):
"""

safe_builtins = {
    'print': print,
    'len': len,
    'sum': sum,
    'max': max,
    'min': min,
    'round': round,
    'str': str,
    'int': int,
    'float': float,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'range': range,
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'repr': repr,
}

def get_answer_from_data(user_query: str, history: list[dict[str, str]]) -> dict:
    formatted_history = ""
    for message in history:
        if message['sender'] == 'user':
            formatted_history += f"User: {message['text']}\n"
        else:
            formatted_history += f"Assistant: {message['text']}\n"
    
    prompt = USER_PROMPT_TEMPLATE.format(
        schema=df_schema,
        chat_history=formatted_history,
        question=user_query
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.0
            )
        )
        generated_code = response.text.strip()
        
        generated_code = generated_code.replace('```python', '').replace('```', '').strip()
        
        print(f"Generated code:\n{generated_code}")
        
        local_vars = {"df": df.copy(), "pd": pd, "json": json}
        
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        try:
            exec(generated_code, {"__builtins__": safe_builtins}, local_vars)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error executing generated code: {e}")
            print(f"Code was:\n{generated_code}")
            return {'answer': f"Error analyzing data: {e}", 'chart': None}
        
        sys.stdout = old_stdout
        output = redirected_output.getvalue().strip()
        
        if not output:
            return {'answer': "The query ran but produced no output.", 'chart': None}

        try:
            result = json.loads(output)
            if isinstance(result, dict):
                return {
                    'answer': result.get('answer', 'Query executed.'),
                    'chart': result.get('chart')
                }
            else:
                return {'answer': str(result), 'chart': None}
        except json.JSONDecodeError:
            print(f"Output was not valid JSON: {output}")
            return {'answer': output, 'chart': None}

    except Exception as e:
        return {'answer': f"An error occurred with the AI model: {e}", 'chart': None}