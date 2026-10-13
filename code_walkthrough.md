# Evonence AI: Line-by-Line Code Walkthrough
## Every Line Explained in Interview-Ready Simple Language

This document walks through **every significant line of code** across all files in the `code/` folder. Each explanation is concise enough to speak aloud confidently in an interview, without being overly academic.

---

# 📄 FILE 1: `code/requirements.txt`

```
pandas
openpyxl
python-dotenv
boto3
pydantic
langchain
langchain-aws
langgraph
flask
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `pandas` | A Python library for reading, storing, and processing tabular data. | We use it to read the Excel file of tweets and run exact percentage calculations. |
| `openpyxl` | A library that lets Pandas open and read `.xlsx` Excel files. | Without it, `pd.read_excel()` would fail to parse Microsoft Excel format. |
| `python-dotenv` | Reads key-value pairs from a `.env` file and injects them into environment variables. | We use it to load our secret AWS credentials securely without hardcoding them in Python. |
| `boto3` | The official Amazon Web Services (AWS) Python SDK. | We use it to create a Bedrock client that sends our prompts to Amazon Nova Pro. |
| `pydantic` | A Python data validation library that enforces strict schemas using Python type hints. | We use it to define the `SentimentAnalysisResult` schema and force the AI to return structured JSON. |
| `langchain` | The core LangChain framework providing base classes like `BaseMessage`, `BaseTool`, and the Runnable protocol. | It is the foundation that connects the LLM, the tools, and the agent together. |
| `langchain-aws` | The LangChain partner package that provides `ChatBedrock`, the Bedrock-specific chat model wrapper. | It handles converting LangChain message objects into Bedrock's Converse API format automatically. |
| `langgraph` | The agent orchestration framework that manages the ReAct reasoning loop and tool dispatch. | It is what makes the system an actual agent with autonomous decision-making instead of a simple LLM call. |
| `flask` | A lightweight Python web framework for building HTTP APIs and serving HTML pages. | It exposes the agent to the browser via the `/api/chat` endpoint and serves the chat UI. |

---

# 📄 FILE 2: `code/main.py` — The Core Agent Engine

## Section 1: Imports

```python
import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `import os` | Imports Python's built-in operating system module. | We use `os.getenv()` to read environment variables and `os.path.join()` to build file paths safely. |
| `import pandas as pd` | Imports the Pandas data analysis library, aliased as `pd`. | Lets us load the Excel file, store tweet data in memory, and calculate percentages with one line of code. |
| `import boto3` | Imports the AWS SDK. | We use `boto3.client('bedrock-runtime')` to establish a secure connection to AWS Bedrock. |
| `from dotenv import load_dotenv` | Imports the function that reads the `.env` file. | Without calling this, `os.getenv()` would return `None` for our AWS credentials. |
| `from pydantic import BaseModel, Field` | Imports the base class and field descriptor for defining data schemas. | `BaseModel` lets us create `SentimentAnalysisResult`; `Field` adds descriptions that guide the AI. |
| `from typing import Literal` | Imports the `Literal` type hint from Python's standard typing module. | `Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]` forces the AI to only choose from these three exact strings. |
| `from langchain_aws import ChatBedrock` | Imports the LangChain wrapper for the Bedrock chat model. | It translates standard LangChain message objects into the exact JSON format Amazon Nova Pro expects. |
| `from langchain_core.tools import tool` | Imports the `@tool` decorator from LangChain's core module. | When we apply `@tool` to a Python function, LangChain converts it into a structured tool the AI can select and call. |
| `from langgraph.prebuilt import create_react_agent` | Imports LangGraph's built-in ReAct agent factory. | This creates the complete reasoning-and-acting loop without us writing the graph manually. |

---

## Section 2: Loading Credentials

```python
# 1. Securely load credentials from .env file
load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not access_key or not secret_key:
    raise ValueError("AWS credentials are missing from the .env file!")
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `load_dotenv()` | Scans for a `.env` file in the current directory and loads its key-value pairs into the process's environment. | Must be called before `os.getenv()` otherwise credential keys return `None`. |
| `access_key = os.getenv("AWS_ACCESS_KEY_ID")` | Reads the AWS Access Key from the environment variables into a Python string. | This is the username part of our AWS credential pair. |
| `secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")` | Reads the secret key into a Python string. | This is the password part of our AWS credential pair. |
| `if not access_key or not secret_key:` | Checks if either credential is `None` or an empty string. | Catches configuration errors immediately at startup instead of failing during the first API call minutes later. |
| `raise ValueError("AWS credentials are missing...")` | Raises a built-in Python exception that halts execution with a clear message. | Fail-fast principle: better to crash loudly at startup than silently fail mid-conversation. |

---

## Section 3: AWS Bedrock Client & Model Setup

```python
# 2. Initialize the Bedrock client exactly as you tested in test_bedrock.py
boto3_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1', 
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

model_id = 'amazon.nova-pro-v1:0'

# Initialize LangChain with ChatBedrock
llm = ChatBedrock(
    client=boto3_client,
    model_id=model_id,
    model_kwargs=dict(temperature=0.1)
)
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `boto3.client('bedrock-runtime', ...)` | Creates a low-level AWS API client connected to the Bedrock Runtime service. | This is the HTTP connection object that sends our prompts to the Amazon cloud. |
| `region_name='us-east-1'` | Tells `boto3` to connect to Bedrock's servers in the US East (N. Virginia) region. | Amazon Nova Pro was not available for on-demand inference in our default region (`ap-south-1`), so we explicitly target `us-east-1` where it is fully supported. |
| `model_id = 'amazon.nova-pro-v1:0'` | A string constant identifying the exact Amazon model we want to use. | Bedrock hosts many models; this ID uniquely selects Amazon Nova Pro version 1 for our reasoning tasks. |
| `ChatBedrock(client=boto3_client, model_id=model_id, ...)` | Wraps the raw `boto3` client in a LangChain-compatible chat model object. | Without this wrapper, we would have to manually format JSON request payloads and parse byte-stream responses for every single API call. |
| `model_kwargs=dict(temperature=0.1)` | Passes the temperature parameter to the model's inference configuration. | A temperature of 0.1 keeps the AI's responses very consistent and close to the most probable answer, which is critical for reliable tool selection. |

---

## Section 4: Pydantic Schema — `SentimentAnalysisResult`

```python
# 3. Define the Strict Output Schema enforcing Chain of Contradiction
class SentimentAnalysisResult(BaseModel):
    pragmatic_analysis: str = Field(description="Analysis of fact vs opinion and implicit context.")
    surface_sentiment: str = Field(description="The literal sentiment of the words.")
    true_intention: str = Field(description="The actual intent of the speaker.")
    contradiction_flag: bool = Field(description="True if surface sentiment contradicts true intention.")
    final_classification: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(description="The final strict categorical classification.")
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `class SentimentAnalysisResult(BaseModel):` | Creates a new Python class that inherits from Pydantic's `BaseModel`, making it a validated schema. | This class becomes the "contract" for what the AI must return — every call must produce a Python object matching these exact fields. |
| `pragmatic_analysis: str` | Declares a required string field for the AI's contextual analysis of the text. | Forces the AI to first explain the real-world context of the tweet before jumping to a conclusion, preventing shallow classifications. |
| `surface_sentiment: str` | Declares a field for the literal word-level sentiment. | Captures what the words say on the surface — important for detecting sarcasm when compared against `true_intention`. |
| `true_intention: str` | Declares a field for the speaker's actual underlying meaning. | This is where sarcasm becomes detectable — if a positive surface contradicts a negative intention, sarcasm is present. |
| `contradiction_flag: bool` | A `True`/`False` field indicating whether surface and true sentiment disagree. | When `True`, it signals the agent that sarcasm was detected and the surface label should be overridden. |
| `final_classification: Literal[...]` | Restricts the output to exactly one of three allowed strings. | Prevents the AI from returning anything other than `POSITIVE`, `NEGATIVE`, or `NEUTRAL`, so our DataFrame column is always clean. |

---

## Section 5: `preprocess_social_dataset()` Function

```python
def preprocess_social_dataset(file_path: str) -> pd.DataFrame:
    print("Pre-processing dataset using Claude 4.6 Sonnet structured outputs...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Warning: Could not load excel file: {e}")
        return pd.DataFrame(columns=["Tweet_ID", "Tweet_Text", "sentiment", "agent_reasoning"])
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `def preprocess_social_dataset(file_path: str) -> pd.DataFrame:` | Defines a function that takes a file path and returns an enriched Pandas DataFrame. | Encapsulating the ingestion logic in a function keeps it reusable and testable. |
| `try: df = pd.read_excel(file_path)` | Attempts to load the `.xlsx` file into a Pandas DataFrame. | Pandas automatically handles column naming and data type detection for Excel files. |
| `except Exception as e:` | Catches any error during file loading (missing file, corrupted file, wrong path). | Prevents a startup crash if the dataset file is accidentally moved or renamed. |
| `return pd.DataFrame(columns=[...])` | Returns an empty DataFrame with the correct column structure as a safe fallback. | The agent's tools can still operate without crashing even if the dataset fails to load. |

```python
    df['sentiment'] = 'NEUTRAL'
    df['agent_reasoning'] = ''
    
    system_instruction = """
    You are an expert linguistic analyst. Classify text strictly as POSITIVE, NEGATIVE, or NEUTRAL.
    Differentiate factual reporting (Neutral) from opinionated sentiment. 
    Detect sarcasm by checking for contradictions between literal words and true intent.
    """
    
    # Langchain natively supports Pydantic structured outputs with Bedrock Claude models!
    structured_llm = llm.with_structured_output(SentimentAnalysisResult)
    
    print(f"Found {len(df)} tweets. Processing the first 5 for demonstration...")
    demo_df = df.head(5).copy()
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `df['sentiment'] = 'NEUTRAL'` | Adds a new column to the DataFrame and fills every row with `'NEUTRAL'` as a default. | Pre-allocating columns ensures the DataFrame has the right structure even if some rows fail to process later. |
| `df['agent_reasoning'] = ''` | Adds a second column for the reasoning string and fills it with empty strings. | Same reason — safe default that prevents `KeyError` when accessing the column later. |
| `system_instruction = """..."""` | Defines the multi-line instruction string passed to the AI as its role and rules. | This is prompt engineering — giving the AI a clear persona and task definition before showing it any tweet. |
| `structured_llm = llm.with_structured_output(SentimentAnalysisResult)` | Creates a new version of the `ChatBedrock` model bound to our Pydantic schema. | When we call this, instead of returning a raw text response, it returns an instantiated `SentimentAnalysisResult` Python object with validated fields. |
| `demo_df = df.head(5).copy()` | Takes only the first 5 rows of the full 100-row dataset into a new DataFrame copy. | Processing 100 tweets sequentially would take ~2 minutes on startup; 5 takes ~6 seconds, making live demos instant. `.copy()` prevents Pandas from editing the original DataFrame accidentally. |

```python
    for index, row in demo_df.iterrows():
        tweet_text = row['Tweet_Text']
        try:
            prompt = f"{system_instruction}\n\nAnalyze this text: {tweet_text}"
            structured_data = structured_llm.invoke(prompt)
            
            demo_df.at[index, 'sentiment'] = structured_data.final_classification
            reasoning_string = (
                f"Pragmatic Context: {structured_data.pragmatic_analysis} | "
                f"Sarcasm Check: {'Contradiction detected' if structured_data.contradiction_flag else 'No contradiction'} | "
                f"Conclusion: {structured_data.true_intention}"
            )
            demo_df.at[index, 'agent_reasoning'] = reasoning_string
            print(f"Processed {row['Tweet_ID']}: {structured_data.final_classification}")
        except Exception as e:
            print(f"Failed to process {row['Tweet_ID']}: {e}")
            demo_df.at[index, 'sentiment'] = "NEUTRAL"
            demo_df.at[index, 'agent_reasoning'] = "Error during processing."
            
    return demo_df
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `for index, row in demo_df.iterrows():` | Loops through every row in the 5-row sample DataFrame, giving us the row number (`index`) and the row data (`row`). | We need to call the AI separately for each tweet because a language model analyzes one text at a time. |
| `tweet_text = row['Tweet_Text']` | Extracts the raw tweet string from the current row. | This is the actual customer post we are sending to Amazon Nova Pro for analysis. |
| `prompt = f"{system_instruction}\n\nAnalyze this text: {tweet_text}"` | Builds the full text prompt by combining our instruction with the specific tweet. | The `\n\n` separates the system instructions from the actual data, making it easier for the model to parse. |
| `structured_data = structured_llm.invoke(prompt)` | Sends the prompt to Bedrock, waits for the response, and returns it as a validated `SentimentAnalysisResult` Python object. | This single line handles the entire HTTP API call, response parsing, and Pydantic validation chain. |
| `demo_df.at[index, 'sentiment'] = structured_data.final_classification` | Writes the validated classification label into the correct row and column of our DataFrame. | `.at[index, column]` is the fastest Pandas method for setting a single cell value by index position. |
| `reasoning_string = (f"Pragmatic Context: ... | Sarcasm Check: ... | Conclusion: ...")` | Builds a human-readable explanation string by combining three fields from the Pydantic object. | This reasoning string is what the agent shows users when they ask "Why was TWT-001 classified as Negative?" |
| `except Exception as e:` ... `demo_df.at[index, 'sentiment'] = "NEUTRAL"` | If the Bedrock API fails for any reason (timeout, rate limit, schema error), falls back to `NEUTRAL` and continues. | Defensive programming — one bad API response should not crash the whole startup. |
| `return demo_df` | Returns the enriched 5-row DataFrame to the caller. | This DataFrame becomes `GLOBAL_DATASET_DF`, the in-memory knowledge base for both agent tools. |

---

## Section 6: Running Pre-processing at Startup

```python
# 4. Run Pre-processing Phase
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "Synthetic Data - Social Media Tweets.xlsx")
GLOBAL_DATASET_DF = preprocess_social_dataset(DATA_FILE)
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `os.path.dirname(__file__)` | Returns the absolute directory path of the currently running Python file (`main.py`'s folder). | Anchors the file path to the script's location so it works correctly regardless of where the user launches Python from. |
| `os.path.join(..., "..", "Synthetic Data - ...")` | Navigates one level up from `code/` (using `".."`) and then into the dataset filename. | The Excel file lives at the project root, not inside `code/`, so we need to go up one directory. |
| `GLOBAL_DATASET_DF = preprocess_social_dataset(DATA_FILE)` | Calls the preprocessing function immediately when the Python module is loaded and stores the result as a module-level variable. | Capital letters signal that this is a **module-level global** — it is available to both tool functions without passing it as an argument. |

---

## Section 7: Tool 1 — `calculate_overall_sentiment_distribution()`

```python
# 5. Define Deterministic Tools
@tool
def calculate_overall_sentiment_distribution() -> dict:
    """
    Calculates the macro-level percentage distribution of sentiments across the entire dataset.
    The agent MUST use this tool to answer questions about overall percentages or statistics.
    """
    total_records = len(GLOBAL_DATASET_DF)
    if total_records == 0:
        return {"error": "Dataset is currently empty."}
        
    distribution = GLOBAL_DATASET_DF['sentiment'].value_counts(normalize=True) * 100
    
    return {
        "total_tweets_analyzed": total_records,
        "positive_percentage": round(distribution.get('POSITIVE', 0), 2),
        "negative_percentage": round(distribution.get('NEGATIVE', 0), 2),
        "neutral_percentage": round(distribution.get('NEUTRAL', 0), 2)
    }
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `@tool` | A decorator that registers the function as an AI-callable LangChain tool and auto-generates its JSON schema from the signature and docstring. | Without this decorator, the function is just a regular Python function invisible to the LangGraph agent. |
| `def calculate_overall_sentiment_distribution() -> dict:` | Defines a function that takes no arguments and returns a Python dictionary. | No arguments needed because the data lives in `GLOBAL_DATASET_DF`, a module-level variable the function reads directly. |
| **The docstring** `"""Calculates the macro-level..."""` | A multi-line documentation string that LangChain reads to understand what the tool does. | The AI reads this docstring to decide when to call this tool — the phrase *"MUST use this tool"* is a strong instruction that prevents the AI from trying to calculate math itself. |
| `total_records = len(GLOBAL_DATASET_DF)` | Counts the total number of rows in our pre-processed dataset. | Used as the denominator label in the response and as a guard check for the empty dataset case. |
| `if total_records == 0: return {"error": "Dataset is currently empty."}` | Returns an informative error dictionary if the dataset failed to load. | Prevents a `ZeroDivisionError` and gives the agent something meaningful to tell the user. |
| `GLOBAL_DATASET_DF['sentiment'].value_counts(normalize=True) * 100` | The key mathematical operation. `value_counts()` counts each unique sentiment label; `normalize=True` converts counts to proportions (0.0–1.0); `* 100` converts to percentages. | This is why we never ask the AI to do math — this single deterministic Pandas expression gives us 100% accurate percentages instantly. |
| `distribution.get('POSITIVE', 0)` | Safely retrieves the POSITIVE percentage from the result, returning `0` if no positive tweets were found. | Prevents `KeyError` if a category has zero occurrences and does not appear in the result. |
| `round(..., 2)` | Rounds the floating-point percentage to two decimal places. | Prevents long decimal strings like `33.333333333333336%` in the final response. |

---

## Section 8: Tool 2 — `query_specific_tweet()`

```python
@tool
def query_specific_tweet(tweet_id: str) -> dict:
    """
    Retrieves the specific classification and the pre-computed logical reasoning for a single tweet ID.
    """
    tweet_row = GLOBAL_DATASET_DF[GLOBAL_DATASET_DF['Tweet_ID'] == tweet_id]
    
    if tweet_row.empty:
        return {"error": f"Tweet ID {tweet_id} could not be located in the dataset."}
        
    return {
        "tweet_text": tweet_row.iloc[0]['Tweet_Text'],
        "classification": tweet_row.iloc[0]['sentiment'],
        "agent_reasoning": tweet_row.iloc[0]['agent_reasoning']
    }
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `def query_specific_tweet(tweet_id: str) -> dict:` | Defines a function with one argument, the tweet ID string, returning a dictionary. | The `tweet_id: str` type annotation is read by `@tool` to generate the JSON schema property `"tweet_id": {"type": "string"}`. |
| `GLOBAL_DATASET_DF[GLOBAL_DATASET_DF['Tweet_ID'] == tweet_id]` | Filters the DataFrame to rows where the `Tweet_ID` column exactly matches the requested ID string. | This is a standard Pandas boolean mask filter — it returns a smaller DataFrame containing only the matching row(s). |
| `if tweet_row.empty:` | Checks if the filtered result is empty (no matching rows found). | A non-existent ID like `TWT-999` would return an empty DataFrame; we handle this gracefully instead of throwing an `IndexError`. |
| `return {"error": f"Tweet ID {tweet_id} could not be located..."}` | Returns a structured error dictionary the agent can relay to the user. | The agent reads this as tool output and responds naturally: "That tweet ID was not found in our dataset." |
| `tweet_row.iloc[0]['Tweet_Text']` | Accesses the first matching row (index 0) by position and retrieves the `Tweet_Text` column value. | `.iloc[0]` converts the DataFrame row to a Series so we can access individual column values by name. |
| `tweet_row.iloc[0]['agent_reasoning']` | Retrieves the pre-computed reasoning string that was built during the startup preprocessing phase. | This is the Chain of Contradiction explanation that gives users a detailed linguistic analysis of why the tweet was classified a certain way. |

---

## Section 9: System Prompt

```python
# 6. Construct the Agent
system_prompt = """
# CORE IDENTITY & PERSONA
You are Evonence AI, an elite Enterprise Social Media Intelligence Analyst...

# COGNITIVE PROCESSING & REASONING
When analyzing tweets, you must look beyond the surface...

# STRICT ARCHITECTURAL CONSTRAINTS
- NO HALLUCINATIONS: You are strictly forbidden from calculating percentages...
- DETERMINISTIC TOOLS: If asked for overall distribution or percentages, you MUST use the `calculate_overall_sentiment_distribution` tool.
- SPECIFIC INQUIRIES: If asked about a specific tweet ID (e.g., TWT-001), you MUST use the `query_specific_tweet` tool.

# OUTPUT AESTHETICS & FORMATTING
- Narrative Flow: Write fluid, conversational paragraphs...
- Zero Robotic Footers: NEVER end your response with generic chatbot phrases...
- Zero Tool Exposure: Never mention that you are using tools...
"""
```

| Block | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `# CORE IDENTITY & PERSONA` | Sets the AI's role as a professional analyst presenting to executives. | Prevents robotic or informal responses — the AI adopts this persona for every single answer. |
| `# COGNITIVE PROCESSING & REASONING` | Instructs the AI to detect sarcasm and pragmatic context beyond literal words. | Without this, the AI might give surface-level keyword-based answers. |
| `NO HALLUCINATIONS` constraint | Explicitly forbids the AI from performing mental arithmetic. | This is the most critical safety constraint — without it, the AI might guess percentages rather than calling the tool. |
| `MUST use the tool` instructions | Hardcodes the decision rule for which tool to call and when. | Acts as a programmatic instruction embedded in natural language — prevents the AI from second-guessing itself. |
| `Zero Robotic Footers` | Bans phrases like "Let me know if you need anything else!" | Makes the agent feel like a real executive analyst rather than a generic chatbot. |
| `Zero Tool Exposure` | Instructs the agent never to say "I will now call the tool…". | Preserves the illusion of intelligence — the agent presents facts as its own knowledge, not as API call results. |

---

## Section 10: Creating the Agent

```python
agent_executor = create_react_agent(
    model=llm,
    tools=[calculate_overall_sentiment_distribution, query_specific_tweet]
)

# The agent is exported for use in app.py
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `create_react_agent(model=llm, tools=[...])` | Builds and compiles a LangGraph `CompiledStateGraph` implementing the ReAct loop: the model reasons, selects a tool, the tool runs, the result feeds back to the model, and this repeats until a final answer is produced. | Replaces what would otherwise be hundreds of lines of custom graph code. |
| `model=llm` | Passes our `ChatBedrock` instance as the reasoning brain of the agent. | The agent uses this model for every reasoning step and final response generation. |
| `tools=[calculate_overall_sentiment_distribution, query_specific_tweet]` | Registers our two decorated Python functions as the agent's available action set. | The agent can only call these two tools — nothing else. This is a security boundary. |
| `# The agent is exported for use in app.py` | Comment noting that `agent_executor` and `system_prompt` are imported by `app.py`. | Because Python module-level code runs on import, everything above executes once when `app.py` does `from main import agent_executor`. |

---

# 📄 FILE 3: `code/app.py` — Flask Web Server & Security Layer

## Section 1: Imports & Flask Initialization

```python
import os
import time
from functools import wraps
from flask import Flask, render_template, request, jsonify
from main import agent_executor, system_prompt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `import time` | Imports Python's built-in time module. | We use `time.time()` to get the current Unix timestamp (seconds since 1970), which powers the rate limiter's time window calculation. |
| `from functools import wraps` | Imports the `wraps` decorator utility. | Used inside our `rate_limit` decorator to preserve the original function's name and docstring, which Flask needs for routing to work correctly. |
| `from flask import Flask, render_template, request, jsonify` | Imports Flask's core components: the app class, HTML template renderer, request object, and JSON response converter. | Each does a specific job — `render_template` serves the HTML page; `request` reads incoming JSON; `jsonify` converts Python dicts to HTTP JSON responses. |
| `from main import agent_executor, system_prompt` | Imports the compiled LangGraph agent and system prompt from `main.py`. | This triggers `main.py` to execute fully on import, including preprocessing the tweets. |
| `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` | Gets the absolute folder path of `app.py` itself. | On Windows, `__file__` can return relative paths depending on how you run the script. Using `abspath` guarantees we always get the full path. |
| `template_folder=os.path.join(BASE_DIR, 'templates')` | Tells Flask exactly where to find HTML templates by absolute path. | Without this, Flask searches for templates relative to the current working directory, causing `TemplateNotFound` errors when running from a different folder. |

---

## Section 2: Rate Limiter

```python
# --- SECURITY LAYER: IN-MEMORY RATE LIMITER ---
RATE_LIMIT = 5 # Maximum 5 requests
RATE_WINDOW = 60 # per 60 seconds
ip_records = {}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()
        
        # Cleanup records older than the window
        if client_ip in ip_records:
            ip_records[client_ip] = [t for t in ip_records[client_ip] if current_time - t < RATE_WINDOW]
        else:
            ip_records[client_ip] = []
            
        # Check limit
        if len(ip_records[client_ip]) >= RATE_LIMIT:
            print(f"SECURITY ALERT: Rate limit exceeded by IP {client_ip}")
            return jsonify({"error": "Security Layer: Rate limit exceeded (Max 5 requests/min). Please slow down."}), 429
            
        ip_records[client_ip].append(current_time)
        return f(*args, **kwargs)
    return decorated_function
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `RATE_LIMIT = 5` and `RATE_WINDOW = 60` | Constants defining our policy: 5 requests allowed per 60 seconds per IP. | Using named constants makes the policy easy to change without hunting through code. |
| `ip_records = {}` | Creates an empty dictionary at module level to store request history per IP address. Example: `{"127.0.0.1": [1699000000.0, 1699000005.0]}` | The dictionary key is the client IP; the value is a list of Unix timestamps for each recent request. |
| `def rate_limit(f):` | Defines a **decorator** — a function that takes another function `f` as input and wraps it. | Decorators let us add the rate-limiting behavior to any route function with just one line (`@rate_limit`) without duplicating code. |
| `@wraps(f)` | Copies the original function's name, docstring, and other metadata to the wrapper function. | Flask uses the function name as the route endpoint identifier. Without `@wraps`, all routes would have the same name `decorated_function`, causing a runtime conflict. |
| `client_ip = request.remote_addr` | Reads the client's IP address from the incoming HTTP request. | This is the unique identifier we use to track requests per user. In a production environment behind a reverse proxy, we would read `X-Forwarded-For` instead. |
| `current_time = time.time()` | Gets the current timestamp in floating-point seconds since Unix epoch. | Used to calculate whether each stored timestamp is within the 60-second sliding window. |
| `ip_records[client_ip] = [t for t in ip_records[client_ip] if current_time - t < RATE_WINDOW]` | **Sliding Window Cleanup:** Rebuilds the timestamp list keeping only entries from the last 60 seconds. | This is the core of the sliding window algorithm — old timestamps automatically expire, allowing users to make new requests after the window passes. |
| `if len(ip_records[client_ip]) >= RATE_LIMIT:` | Checks if this IP has already used up all 5 allowed requests within the current 60-second window. | If true, we block the request before it ever reaches the AI — protecting both our AWS costs and server stability. |
| `return jsonify({"error": "..."}), 429` | Returns an HTTP 429 ("Too Many Requests") response with a JSON error message. | HTTP 429 is the industry-standard status code for rate limiting. The frontend JavaScript reads this and shows it as an error message. |
| `ip_records[client_ip].append(current_time)` | Records the current timestamp for this IP, counting the request that is about to be processed. | Must happen after the limit check and before calling `f()` to accurately track request times. |
| `return f(*args, **kwargs)` | Calls the original wrapped function (our `/api/chat` handler) and returns its result. | This is where the actual request processing continues if the rate check passed. |

---

## Section 3: Security Headers

```python
# --- SECURITY LAYER: HTTP HEADERS ---
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com;"
    return response
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `@app.after_request` | A Flask decorator that registers a function to run on every HTTP response before it is sent to the browser. | This guarantees security headers are added to every single response without duplicating code in each route. |
| `X-Content-Type-Options: nosniff` | Tells the browser to strictly follow the declared Content-Type and never guess the file type. | Prevents an attacker from uploading a malicious script disguised as an image and having the browser execute it. |
| `X-Frame-Options: DENY` | Completely blocks the page from being loaded inside an HTML `<iframe>`. | Prevents **Clickjacking attacks**, where an attacker embeds our chat page invisibly on their site and tricks users into clicking our buttons. |
| `X-XSS-Protection: 1; mode=block` | Activates the browser's built-in Cross-Site Scripting filter and tells it to block the page if an attack is detected. | Adds an extra browser-level XSS defense, though modern browsers rely on CSP for this. |
| `Content-Security-Policy` | Declares a whitelist of trusted sources for scripts, styles, and fonts. | The most powerful web security header — it tells the browser to reject any resource not from our server or the approved CDNs, completely blocking injected malicious scripts. |

---

## Section 4: Routes & Chat Endpoint

```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
@rate_limit # Applied Security Rate Limiting
def chat():
    user_input = request.json.get('message', '')
    
    # --- SECURITY LAYER: INPUT VALIDATION ---
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    if len(user_input) > 500:
        print(f"SECURITY ALERT: Payload too large from IP {request.remote_addr}")
        return jsonify({"error": "Security Layer: Payload too large. Maximum 500 characters allowed."}), 413
        
    try:
        response = agent_executor.invoke({"messages": [("system", system_prompt), ("user", user_input)]})
        bot_reply = response['messages'][-1].content
        return jsonify({"response": bot_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Enterprise Sentiment Agent Web UI on port 5000...")
    app.run(debug=True, port=5000)
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `@app.route('/')` | Registers the `index()` function as the handler for GET requests to the root URL `/`. | When a user visits `http://127.0.0.1:5000/`, Flask calls this function. |
| `return render_template('index.html')` | Reads and returns the `templates/index.html` file as an HTTP HTML response. | This sends the chat interface to the user's browser. |
| `@app.route('/api/chat', methods=['POST'])` | Registers the `/api/chat` URL but only accepts HTTP POST requests. | GET would expose the endpoint to browser URL bar access; POST enforces that data is sent in the request body. |
| `@rate_limit` | Applies our custom rate-limiting decorator to this specific route. | Stacking decorators in Python applies them bottom-up, so `@rate_limit` runs first, and if it passes, `chat()` runs. |
| `user_input = request.json.get('message', '')` | Parses the JSON body of the POST request and extracts the `message` field. | If JavaScript sends `{"message": "What is TWT-001?"}`, this line extracts the string `"What is TWT-001?"`. The `''` default prevents a crash if the key is missing. |
| `if not user_input:` ... `return ..., 400` | Returns HTTP 400 ("Bad Request") if the message is empty or missing. | Prevents sending blank prompts to the Bedrock API, which wastes tokens and costs money. |
| `if len(user_input) > 500:` ... `return ..., 413` | Returns HTTP 413 ("Content Too Large") if the input exceeds 500 characters. | Security measure: limits prompt injection attack surface and prevents excessive token consumption. |
| `agent_executor.invoke({"messages": [("system", system_prompt), ("user", user_input)]})` | Runs the full LangGraph ReAct agent with the system prompt and user message as the initial state. | The agent takes over from here: it reasons, selects a tool, runs it, synthesizes a response, and returns. |
| `response['messages'][-1].content` | Accesses the final element of the messages list and reads its text content. | LangGraph returns all messages (system, user, tool calls, tool results, final AI response) in a list; `[-1]` efficiently gets the last one, which is always the agent's final answer. |
| `return jsonify({"response": bot_reply})` | Converts the Python string into a JSON HTTP response `{"response": "..."}`. | The JavaScript `fetch()` call in the browser reads this JSON and extracts the `response` field to display. |
| `except Exception as e: return jsonify({"error": str(e)}), 500` | Catches any unexpected error from the agent (e.g., Bedrock timeout) and returns a 500 error. | Without this, an unhandled exception would return an ugly HTML traceback to the browser instead of a clean JSON error. |
| `if __name__ == '__main__': app.run(debug=True, port=5000)` | Only runs the Flask development server if this file is executed directly (not imported). | `debug=True` enables auto-reloading on code changes; `port=5000` sets the port the server listens on. |

---

# 📄 FILE 4: `code/templates/index.html` — Chat Interface

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Sentiment Analysis Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `<!DOCTYPE html>` | Tells the browser this is a modern HTML5 document. | Without this, browsers enter "quirks mode" which renders CSS differently and inconsistently. |
| `<meta charset="UTF-8">` | Sets character encoding to UTF-8, supporting all international characters and emojis. | Our tweet data may contain emojis, special characters, and non-ASCII text that would display incorrectly otherwise. |
| `<meta name="viewport" ...>` | Tells mobile browsers not to scale down the page and to use the device's actual width. | Ensures the chat interface renders correctly on phones and tablets. |
| `<title>Enterprise Sentiment Analysis Agent</title>` | Sets the text shown in the browser tab. | Professional presentation detail for the hackathon demo. |
| `<link href="https://fonts.googleapis.com/css2?family=Inter...">` | Loads the Inter typeface from Google's CDN. | Inter is a clean, professional sans-serif font designed for UI screens. Using a CDN avoids hosting font files locally. |
| `{{ url_for('static', filename='style.css') }}` | A **Jinja2 template tag** that Flask replaces with the correct URL path to `style.css`. | Using `url_for` instead of hardcoding `/static/style.css` ensures the path works correctly regardless of the Flask URL prefix or deployment environment. |

```html
<body>
    <div class="bg-gradient"></div>
    <div class="app-container">
        <header class="header glass-panel">
            <div class="logo">
                <span class="logo-icon">✨</span>
                <h1>Evonence AI</h1>
            </div>
            <div class="status-badge">
                <span class="pulse"></span> Amazon Nova Pro Active
            </div>
        </header>

        <main class="chat-container glass-panel">
            <div class="chat-history" id="chat-history">
                <div class="message bot">
                    <div class="message-avatar">🤖</div>
                    <div class="message-bubble">
                        Hello! I'm the Enterprise Sentiment Analysis Agent...
                    </div>
                </div>
            </div>
            <div class="input-area">
                <form id="chat-form">
                    <input type="text" id="user-input" placeholder="Ask about the dataset..." autocomplete="off">
                    <button type="submit" id="send-btn">
                        <svg ...></svg>
                    </button>
                </form>
            </div>
        </main>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `<div class="bg-gradient">` | An empty div that CSS uses as a canvas for the animated background gradient. | Keeps background animation separate from content HTML — a clean separation of concerns. |
| `class="glass-panel"` | Applies the CSS glassmorphism styling (frosted-glass blur effect). | Gives the header and chat container a premium dark, translucent appearance. |
| `<span class="pulse"></span> Amazon Nova Pro Active` | A green pulsing dot followed by text indicating the AI model is online. | Shows the user that the system is connected and running live, similar to an "Online" indicator in professional dashboards. |
| `id="chat-history"` | A unique ID on the chat messages container. | `script.js` uses `document.getElementById('chat-history')` to find this element and append new messages dynamically. |
| `autocomplete="off"` | Disables browser autocomplete suggestions on the input field. | Prevents the browser from suggesting previous entries, which would be irrelevant and distracting in a technical query context. |
| `marked/marked.min.js` (CDN) | Loads the Marked.js Markdown parser from jsDelivr's CDN. | The AI's responses contain Markdown formatting (bold, bullets, headers). This library converts the Markdown syntax into rendered HTML. |
| Scripts at the **bottom** of `<body>` | Both scripts are placed at the end of the HTML body. | Scripts at the bottom load after the HTML structure is built, ensuring elements like `#chat-form` and `#user-input` already exist when JavaScript tries to find them. |

---

# 📄 FILE 5: `code/static/script.js` — Client-Side Behavior

```javascript
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `document.addEventListener('DOMContentLoaded', ...)` | Wraps all code in an event listener that fires only after the browser has finished building the HTML DOM. | Guarantees that HTML elements like `#chat-form` and `#user-input` exist before JavaScript tries to access them. |
| `document.getElementById('chat-form')` | Gets a direct JavaScript reference to the HTML form element by its unique `id`. | Caching the element in a `const` variable means we look it up once instead of on every interaction, which is more efficient. |

```javascript
    const scrollToBottom = () => {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    const formatText = (text) => {
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
    };
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `chatHistory.scrollTop = chatHistory.scrollHeight` | Sets the scroll position to the maximum, which is the very bottom of the chat. | After adding a new message, this auto-scrolls the chat panel so the user always sees the latest message without manually scrolling. |
| `if (typeof marked !== 'undefined')` | Checks if the Marked.js library successfully loaded from the CDN before trying to use it. | If the CDN fails (network issue, CDN outage), this check prevents a JavaScript crash and falls to the regex fallback below. |
| `marked.parse(text)` | Converts a Markdown string into an HTML string using the Marked.js library. | Turns `**Bold Text**` into `<strong>Bold Text</strong>` and processes the AI's full Markdown formatting. |
| `text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')` | A simple regex fallback that manually converts bold Markdown syntax to HTML bold tags. | Ensures at least basic formatting works even if the CDN script fails to load. |

```javascript
    const addMessage = (text, isUser = false) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        msgDiv.innerHTML = `
            <div class="message-avatar">${isUser ? '👤' : '🤖'}</div>
            <div class="message-bubble">${formatText(text)}</div>
        `;

        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    };
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `document.createElement('div')` | Creates a new, empty HTML `<div>` element in memory (not yet visible on screen). | We build the message structure in memory before adding it to the page for better performance. |
| `` msgDiv.className = `message ${isUser ? 'user' : 'bot'}` `` | Sets the CSS class to either `"message user"` or `"message bot"` using a ternary operator. | The CSS uses these classes to right-align user messages and left-align bot messages with different bubble colors. |
| `msgDiv.innerHTML = \`...\`` | Sets the inner HTML of our new div using a template literal containing the avatar and bubble structure. | Dynamically builds the visual chat bubble with the correct avatar icon and formatted text content. |
| `chatHistory.appendChild(msgDiv)` | Adds the completed message div into the chat history container, making it visible on screen. | This is the line that actually renders the new message in the browser. |
| `scrollToBottom()` | Immediately scrolls the chat container to show the new message. | Called after every append so the user always sees the most recent message. |

```javascript
    const createTypingIndicator = () => {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing';
        typingDiv.id = 'typing-indicator-msg';
        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                <div class="typing-indicator" style="display: flex;">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(typingDiv);
        scrollToBottom();
    };

    const removeTypingIndicator = () => {
        const indicator = document.getElementById('typing-indicator-msg');
        if (indicator) { indicator.remove(); }
    };
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `typingDiv.id = 'typing-indicator-msg'` | Gives the typing indicator element a unique ID. | Lets `removeTypingIndicator()` find and remove exactly this element after the AI response arrives. |
| `<span></span><span></span><span></span>` | Three empty spans styled by CSS to create the animated bouncing dots. | The CSS `@keyframes bounce` animation makes these three dots pulse up and down in sequence, indicating "thinking". |
| `if (indicator) { indicator.remove(); }` | Checks the indicator exists before removing it. | Defensive check — if the response comes back before the indicator was added (extremely fast response), this prevents a `null.remove()` error. |

```javascript
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const message = userInput.value.trim();
        if (!message) return;

        userInput.value = '';
        userInput.disabled = true;

        addMessage(message, true);
        createTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            removeTypingIndicator();

            if (response.ok) {
                addMessage(data.response);
            } else {
                addMessage(`Error: ${data.error}`);
            }
        } catch (error) {
            removeTypingIndicator();
            addMessage('Network error. Could not reach the server.');
        } finally {
            userInput.disabled = false;
            userInput.focus();
        }
    });
```

| Line | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `chatForm.addEventListener('submit', async (e) => {` | Listens for the form submission event and registers an **async** callback. | `async` lets us use `await` inside the function, which pauses execution to wait for the Bedrock API response without blocking the browser. |
| `e.preventDefault()` | Cancels the browser's default form submit behavior (which would reload the page). | Prevents the page from refreshing — we handle the request via JavaScript fetch instead. |
| `userInput.value.trim()` | Gets the text from the input field and removes leading/trailing whitespace. | Prevents submitting a message that is only spaces, which would be a waste of an API call. |
| `userInput.disabled = true` | Disables the input field during the API call. | Prevents the user from sending multiple concurrent requests while waiting for a response. |
| `await fetch('/api/chat', {method: 'POST', body: JSON.stringify({message})})` | Sends an asynchronous HTTP POST request to our Flask backend with the message as a JSON body. | `await` pauses this function until the server responds, so subsequent lines can safely access the response. |
| `headers: { 'Content-Type': 'application/json' }` | Tells the server that the request body is JSON-encoded data. | Without this header, Flask's `request.json` would return `None` and the server would reject the message. |
| `await response.json()` | Parses the HTTP response body as JSON into a JavaScript object. | Converts the server's `{"response": "..."}` string into an accessible JavaScript `data.response` property. |
| `if (response.ok)` | Checks if the HTTP status code is in the 200–299 range (success). | Separates successful responses from error responses (429, 413, 500) so we can display the correct message. |
| `} catch (error) {` | Catches network-level failures (no internet, server down). | Handles the case where `fetch()` itself throws an error (not just an error HTTP status), such as a complete server crash. |
| `} finally { userInput.disabled = false; userInput.focus(); }` | The `finally` block **always** runs, whether the request succeeded or failed. | Guarantees the input field is always re-enabled and focused so the user can type their next question. |

---

# 📄 FILE 6: `code/static/style.css` — Key CSS Concepts

```css
:root {
    --glass-bg: rgba(0, 0, 0, 0.45);
    --glass-border: rgba(255, 255, 255, 0.08);
    --primary: #FFFFFF;
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
}
```

| Concept | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `:root` selector | Targets the top-level HTML element, making these custom properties available globally. | Defines a centralized design system — changing `--glass-bg` here updates every element using that variable. |
| `rgba(0, 0, 0, 0.45)` | Black color at 45% opacity (translucent). | The `a` in `rgba` is the alpha channel — values below 1.0 make the color partially transparent, creating the glass effect. |

```css
.bg-gradient {
    position: absolute;
    background: radial-gradient(circle at 15% 50%, rgba(30, 41, 59, 0.4), transparent 50%), ...;
    animation: drift 20s ease-in-out infinite alternate;
}
```

| Concept | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `position: absolute` + `z-index: -1` | Places the element behind all other content. | The gradient sits as a full-screen background canvas without interfering with the chat layout. |
| `radial-gradient(...)` | Creates a circular gradient that fades from a semi-transparent dark color to fully transparent. | Gives the background a subtle depth effect that feels premium without being distracting. |
| `animation: drift 20s ease-in-out infinite alternate` | Applies the `@keyframes drift` animation, repeating indefinitely and reversing direction each cycle. | The slow 20-second `scale(1.1)` animation makes the background feel alive, adding a subtle cinematic quality. |

```css
.glass-panel {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
}
```

| Concept | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `backdrop-filter: blur(20px)` | Applies a Gaussian blur to everything behind the element, creating the frosted glass effect. | This is the core of the glassmorphism UI style — it gives panels a premium, layered look when placed over the animated background. |
| `border: 1px solid rgba(255, 255, 255, 0.08)` | Adds a very faint white border at 8% opacity. | Defines the edge of the glass panel subtly — too opaque would look harsh; too transparent would be invisible. |

```css
.typing-indicator span {
    animation: bounce 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
```

| Concept | What It Does | Why We Need It |
| :--- | :--- | :--- |
| `animation: bounce 1.4s infinite` | Makes each dot continuously loop through the bounce animation. | The three dots repeatedly scale down and back up, creating the classic "typing..." animation. |
| `animation-delay: -0.32s` and `-0.16s` | Starts the animation at different points in the cycle for each dot. | Negative delays start the animation mid-cycle, creating the staggered wave effect where dots bounce one after another instead of all together. |

---

## Summary Table: Which File Does What

| File | Primary Role | Talks To |
| :--- | :--- | :--- |
| `requirements.txt` | Lists Python packages to install | pip / Python environment |
| `main.py` | LLM client, data ingestion, tools, agent definition | AWS Bedrock, Pandas, LangGraph |
| `app.py` | Web server, security enforcement, routing | `main.py` (imports agent), browser |
| `templates/index.html` | Chat interface HTML structure | CSS file, JavaScript file, Flask template engine |
| `static/script.js` | Browser interaction, API calls, DOM updates | Flask `/api/chat` endpoint, HTML DOM |
| `static/style.css` | Visual styling, animations, glassmorphism theme | HTML element classes and IDs |
