import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from io import StringIO
import sys
import json
import threading
import queue

# --- 1. Load Environment & Configure API ---
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the backend/.env file.")
genai.configure(api_key=api_key)

# --- 2. Load Data ---
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Sugar_Spend_Data.csv')
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found. Make sure 'Sugar_Spend_Data.csv' is in 'backend/app/data/'")

# --- 3. Prepare Context for the LLM ---
buffer = StringIO()
df.info(buf=buffer)
df_schema = buffer.getvalue()

# --- 4. Initialize Model ---
model = genai.GenerativeModel('gemini-2.5.flash')

# --- 5. "KEARNEY LEVEL" SYSTEM INSTRUCTION (Sandbox rules removed) ---
# We now encourage f-strings because the sandbox is gone.
SYSTEM_INSTRUCTION = """
You are a data analysis engine. You will be given a user question and context.
Your ONLY job is to generate a single block of Python code to answer the question.

**CRITICAL: ALL output MUST be a single `print()` call at the end, printing a JSON-formatted string.**

The JSON output *must* have this structure:
{{
    "answer": "Your natural language answer here",
    "chart": null  // or a chart object
}}

**Chart Generation:**
- If the user asks for "Plot the spend for each commodity.":
  `spend = df.groupby('Commodity')['Spend (USD)'].sum()`
  `chart_data = {{"type": "bar", "labels": spend.index.tolist(), "data": spend.values.tolist()}}`
  `print(json.dumps({{"answer": "Here is the spend by commodity.", "chart": chart_data}}))`
- The chart object must be: {{"type": "bar|pie", "labels": [...], "data": [...]}}

**Normal Questions (Data, Lists, Numbers):**
- **Use f-strings for formatting.**
- If the user asks for "list out the commodities":
  `clist = df['Commodity'].unique().tolist()`
  `answer = f"The commodities are: {{', '.join(clist)}}"`
  `print(json.dumps({{"answer": answer, "chart": null}}))`
- If the user asks "What is the total spend?":
  `total_spend = df['Spend (USD)'].sum()`
  `answer = f"The total spend is ${{total_spend:,.2f}}"`
  `print(json.dumps({{"answer": answer, "chart": null}}))`
- **BAD Example:** `print(df['Commodity'].unique().tolist())`  <-- THIS WILL CRASH THE SYSTEM.

**Error Handling:**
- **No Data:** If a filter results in no data, state that in the answer.
  `print(json.dumps({{"answer": "No data was found for 'Honey'.", "chart": null}}))`
- **Vague/Illogical:** If the query is vague or illogical, state that.
  `print(json.dumps({{"answer": "That query is illogical. Please rephrase.", "chart": null}}))`
"""

# --- 6. USER PROMPT TEMPLATE ---
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

# --- 7. "Safe" Execution Environment ---
# The safe_builtins dictionary is REMOVED.
# We will use __builtins__ to give access to all standard functions (like f-strings)
# and rely on a thread-based timeout for security against infinite loops.

# --- 8. Core Query Function ---
def execute_code_in_thread(code, local_vars, result_queue):
    """Target function to run exec in a separate thread."""
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    
    try:
        # Give access to all built-ins (for f-strings, etc.)
        # This is safe because __import__ is not included by default in exec's builtins
        exec(code, {"__builtins__": __builtins__}, local_vars)
        output = redirected_output.getvalue().strip()
        result_queue.put(('success', output))
    except Exception as e:
        result_queue.put(('error', e, code))
    finally:
        sys.stdout = old_stdout

def get_answer_from_data(user_query: str, history: list[dict[str, str]]) -> dict:
    """
    Uses Gemini to generate and execute Python code to answer a question about the dataframe.
    Returns a dictionary: {'answer': ..., 'chart': ...}
    """
    
    # 1. Format the chat history
    formatted_history = ""
    for message in history:
        if message['sender'] == 'user':
            formatted_history += f"User: {message['text']}\n"
        else:
            formatted_history += f"Assistant: {message['text']}\n"
    
    # 2. Construct the prompt
    prompt = USER_PROMPT_TEMPLATE.format(
        schema=df_schema,
        chat_history=formatted_history,
        question=user_query
    )
    
    # 3. Send prompt to Gemini with System Instruction
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.0
            ),
            system_instruction=SYSTEM_INSTRUCTION
        )
        generated_code = response.text.strip()
        
        # 4. Safely execute the generated code with a TIMEOUT
        local_vars = {"df": df.copy(), "pd": pd, "json": json}
        result_queue = queue.Queue()
        
        execution_thread = threading.Thread(
            target=execute_code_in_thread,
            args=(generated_code, local_vars, result_queue)
        )
        
        execution_thread.start()
        # --- TIMEOUT SET TO 3 SECONDS ---
        execution_thread.join(timeout=3.0) 

        if execution_thread.is_alive():
            # Thread is still running - this is an infinite loop or long query
            print("CRITICAL: Code execution timed out. Possible infinite loop.")
            print(f"Code was:\n{generated_code}")
            return {'answer': "Error: Your query took too long to execute and was terminated.", 'chart': None}
        
        # 5. Process the output from the thread
        status, *result = result_queue.get()
        
        if status == 'error':
            e, code = result
            print(f"Error executing generated code: {e}")
            print(f"Code was:\n{code}")
            return {'answer': f"Error analyzing data: {e}", 'chart': None}
        
        output = result[0]
        
        if not output:
            return {'answer': "The query ran but produced no output.", 'chart': None}

        try:
            # The prompt *requires* the output to be a JSON string.
            result_data = json.loads(output)
            if isinstance(result_data, dict):
                return {
                    'answer': result_data.get('answer', 'Query executed.'),
                    'chart': result_data.get('chart')
                }
            else:
                return {'answer': str(result_data), 'chart': None}
        except json.JSONDecodeError:
            print(f"CRITICAL: AI violated prompt. Output was not valid JSON: {output}")
            return {'answer': f"An error occurred. The AI's response was not valid JSON.\nRaw output: {output}", 'chart': None}

    except Exception as e:
        return {'answer': f"An error occurred with the AI model: {e}", 'chart': None}

