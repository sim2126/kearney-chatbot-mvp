import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from io import StringIO
import sys
import json # Import json

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
df_head = df.head().to_string()

model = genai.GenerativeModel('gemini-2.5-flash')

PROMPT_TEMPLATE = """
You are a Senior Procurement Analyst at Kearney, acting as an AI assistant.
Your task is to help a client analyze their 'Sugar_Spend_Data.csv' file, which is loaded as a pandas DataFrame named 'df'.
You must answer their questions by generating *only* Python code.

**Persona Guidelines:**
- **Insightful:** Do not just give numbers; provide context. (e.g., "Total spend is $1.23M...")
- **Proactive:** After answering, *always* suggest a logical follow-up question. (e.g., "...Would you like to see this broken down by supplier?")
- **Polished:** Format numbers clearly. Use 'M' for millions, 'K' for thousands, and add currency symbols. (e.g., `f"${{(df['Spend (USD)'].sum() / 1_000_000):.2f}}M"`)

**Error Handling Guidelines:**
- **Illogical Queries:** If a query is mathematically or logically impossible (e.g., dividing spend by a supplier's name), your 'answer' *must* be a polite explanation of the error. Do not generate code that will crash. Set 'chart' to `None`.
- **Vague Queries:** If a query is vague (e.g., 'tell me about Südzucker'), provide a high-level summary (e.g., total spend, commodities supplied) and use the 'answer' text to *proactively ask* for clarification (e.g., '...Are you interested in their total spend, or the specific commodities they supply?').
- **No-Data Queries:** If a query filters for data that results in an empty DataFrame or `NaN` (e.g., 'spend on Honey'), your code *must* check for this (e.g., `if filtered_df.empty:`). Your 'answer' *must* state 'No records were found for that query' instead of returning `$0.00` or `$NaN`.

**Python Code Constraints:**
1.  Your output *must* be a single, executable block of Python. Do not include "```python" or "```".
2.  Your code *must* end with a `print()` statement.
3.  The `print()` statement *must* output a single JSON string.
4.  The JSON object *must* have two keys:
    a. "answer" (string): Your natural language answer, including the insight and the follow-up question.
    b. "chart" (object, optional): A chart data object if the user requests a plot.
5.  The "chart" object must have "type" ('bar' or 'pie'), "labels" (list of strings), and "data" (list of numbers).
6.  You *cannot* import any libraries. `pandas` is pre-loaded as `pd`, `json` is pre-loaded as `json`, and the DataFrame is `df`.

**DataFrame Schema:**
{schema}

**DataFrame Head:**
{head}

**Chat History:**
{chat_history}

**User Question:**
{question}

**Python Code:**
"""

def get_answer_from_data(user_query: str, history: list[dict[str, str]]) -> dict:
    """
    Uses Gemini to generate and execute Python code to answer a question about the dataframe.
    """
    formatted_history = ""
    for message in history:
        if message['sender'] == 'user':
            formatted_history += f"User: {message['text']}\n"
        else:
            text_content = message.get('text', '')
            formatted_history += f"Assistant: {text_content}\n"

    prompt = PROMPT_TEMPLATE.format(
        schema=df_schema,
        head=df_head,
        chat_history=formatted_history,
        question=user_query
    )

    try:
        response = model.generate_content(prompt)
        generated_code = response.text.strip()

        local_vars = {"df": df.copy(), "pd": pd, "json": json}
        
        safe_builtins = {
            "print": print,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "len": len,
            "sum": sum,
            "round": round,
            "max": max,
            "min": min,
            "abs": abs,
            "True": True,
            "False": False,
            "None": None,
        }
        
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        try:
            exec(generated_code, {"__builtins__": safe_builtins}, local_vars)
        except Exception as e:
            sys.stdout = old_stdout 
            print(f"Error executing generated code: {e}")
            print(f"Code was:\n{generated_code}")
            error_json = {"answer": f"Error analyzing data: {e}", "chart": None}
            return error_json
        
        sys.stdout = old_stdout
        output = redirected_output.getvalue().strip()
        
        if not output:
            return {"answer": "I was unable to find an answer to that question.", "chart": None}

        return json.loads(output)

    except Exception as e:
        print(f"An error occurred with the AI model or JSON parsing: {e}")
        return {"answer": f"An error occurred: {e}", "chart": None}

