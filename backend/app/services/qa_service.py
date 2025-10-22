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
You are a data analysis engine. You will be given a user question and context.
Your ONLY job is to generate a single block of Python code to answer the question.

**CRITICAL: ALL output MUST be a single `print()` call at the end, printing a JSON-formatted string.**
**CRITICAL: Do NOT use f-strings (f"...") as they are not allowed. Use simple string concatenation (+) and str().**

The JSON output *must* have this structure:
{{
    "answer": "Your natural language answer here",
    "chart": null
}}

**Chart Generation:**
- If the user asks for "Plot the spend for each commodity.":
  `spend = df.groupby('Commodity')['Spend (USD)'].sum()`
  `chart_data = {{"type": "bar", "labels": spend.index.tolist(), "data": spend.values.tolist()}}`
  `print(json.dumps({{"answer": "Here is the spend by commodity.", "chart": chart_data}}))`
- If the user asks for a "plot", "chart", or "graph", generate a chart object.
- The chart object must be: {{"type": "bar|pie", "labels": [...], "data": [...]}}

**Normal Questions (Data, Lists, Numbers):**
- **Do NOT use f-strings.** Use `str()` and `round()` for formatting.
- If the user asks for "list out the commodities":
  `clist = df['Commodity'].unique().tolist()`
  `answer = "The commodities are: " + ", ".join(clist)`
  `print(json.dumps({{"answer": answer, "chart": null}}))`
- If the user asks "What is the total spend?":
  `total_spend = df['Spend (USD)'].sum()`
  `answer = "The total spend is $" + str(round(total_spend, 2))`
  `print(json.dumps({{"answer": answer, "chart": null}}))`
- **BAD Example:** `print(df['Commodity'].unique().tolist())`
- **BAD Example:** `answer = f"..."`

**Error Handling:**
- **No Data:** If a filter results in no data, state that in the answer.
  `print(json.dumps({{"answer": "No data was found for 'Honey'.", "chart": null}}))`
- **Vague/Illogical:** If the query is vague or illogical, state that.
  `print(json.dumps({{"answer": "That query is illogical. Please rephrase.", "chart": null}}))`
"""

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=SYSTEM_INSTRUCTION
)

USER_PROMPT_TEMPLATE = """
**Available Data:**
- DataFrame 'df' (pandas as 'pd')
- 'json' module is pre-imported and available.

**Schema:**
{schema}

**Chat History:**
{chat_history}

**User Question:**
{question}

**Python Code (JSON output only):**
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
            print(f"CRITICAL: AI violated prompt. Output was not valid JSON: {output}")
            return {'answer': f"An error occurred. The AI's response was not valid JSON.\nRaw output: {output}", 'chart': None}

    except Exception as e:
        return {'answer': f"An error occurred with the AI model: {e}", 'chart': None}