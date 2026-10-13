import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Securely load credentials from .env file
load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not access_key or not secret_key:
    raise ValueError("AWS credentials are missing from the .env file!")

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

# 3. Define the Strict Output Schema enforcing Chain of Contradiction (From Architecture Doc)
class SentimentAnalysisResult(BaseModel):
    pragmatic_analysis: str = Field(description="Analysis of fact vs opinion and implicit context.")
    surface_sentiment: str = Field(description="The literal sentiment of the words.")
    true_intention: str = Field(description="The actual intent of the speaker.")
    contradiction_flag: bool = Field(description="True if surface sentiment contradicts true intention.")
    final_classification: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(description="The final strict categorical classification.")

# File paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "..", "Synthetic Data - Social Media Tweets.xlsx")
CACHE_FILE = os.path.join(BASE_DIR, "enriched_tweets.csv")

def load_dataset() -> pd.DataFrame:
    """
    Smart loader: uses the pre-computed CSV cache when available (instant load),
    otherwise falls back to live Amazon Nova Pro processing.
    """
    # ── FAST PATH: pre-computed cache exists ───────────────────────────────────
    if os.path.exists(CACHE_FILE):
        print("✅ Pre-computed cache found. Loading enriched_tweets.csv instantly...")
        try:
            df = pd.read_csv(CACHE_FILE)
            print(f"   Loaded {len(df)} pre-enriched tweets from cache. Agent is ready.")
            return df
        except Exception as e:
            print(f"⚠️  Cache read error ({e}). Falling back to live processing.")

    # ── SLOW PATH: no cache — run live Nova Pro inference ─────────────────────
    print("⚙️  No cache found. Running live Amazon Nova Pro pre-processing...")
    try:
        df = pd.read_excel(DATA_FILE)
    except Exception as e:
        print(f"❌ Could not load Excel file: {e}")
        return pd.DataFrame(columns=["Tweet_ID", "Tweet_Text", "sentiment", "agent_reasoning"])

    df['sentiment']       = 'NEUTRAL'
    df['agent_reasoning'] = ''

    system_instruction = """
# ROLE & EXPERTISE
You are Dr. Evonence, a world-class computational linguist and enterprise sentiment intelligence expert with 15 years of experience analyzing social media discourse for Fortune 500 companies. You have deep expertise in pragmatics, discourse analysis, and the psychology of online communication.

Your task is to perform a rigorous, multi-dimensional sentiment analysis on a single social media tweet and classify it as exactly one of: POSITIVE, NEGATIVE, or NEUTRAL.

# YOUR ANALYTICAL METHODOLOGY — CHAIN OF CONTRADICTION
You MUST follow these three steps in sequence before producing any classification:

STEP 1 — PRAGMATIC CONTEXT ANALYSIS:
First, understand the real-world context of the tweet. Ask yourself:
- What is the speaker's actual situation?
- Is this a factual statement about an event, or is it the speaker's personal opinion?
- Does the tweet reference a brand, product, service, or public figure, and if so, what is the implied relationship?
- What domain is this tweet from (customer service, politics, product review, personal expression, news reporting)?

STEP 2 — SURFACE vs. TRUE INTENTION COMPARISON:
Analyze the literal words first, then the true intent:
- Surface Sentiment: What do the words literally say? Positive words = positive surface. Negative words = negative surface.
- True Intention: What does the speaker ACTUALLY mean given the context, tone, punctuation, and cultural subtext?
- Contradiction Flag: If surface sentiment is POSITIVE but true intention is NEGATIVE (or vice versa), a CONTRADICTION exists. This is the primary sarcasm detection signal.

STEP 3 — FINAL CLASSIFICATION DECISION:
Apply the following rules strictly in order:
  1. If a CONTRADICTION exists between surface and true intention → classify based on TRUE INTENTION, not surface.
  2. If the tweet is a neutral factual report (news-style, no personal stance, no emotive language) → NEUTRAL.
  3. If the tweet expresses clear personal positive emotions, satisfaction, praise, excitement, or endorsement → POSITIVE.
  4. If the tweet expresses frustration, anger, disappointment, complaint, disgust, or criticism → NEGATIVE.
  5. If genuinely ambiguous with no clear lean → NEUTRAL.

# SPECIFIC RULES FOR DIFFICULT CASES

## Sarcasm Detection Rules:
- Hyperbolic praise for something clearly terrible = NEGATIVE (e.g., "Oh great, my flight got cancelled AGAIN. Fantastic service!")
- Exaggerated negativity that is clearly self-deprecating humor = may be NEUTRAL or POSITIVE
- Keywords like "sure", "right", "great", "wonderful" followed by obviously bad facts = sarcasm → NEGATIVE
- Exclamation marks combined with negative context are a strong sarcasm signal

## Factual Reporting vs. Opinion:
- "The company reported a 10% drop in revenue" = NEUTRAL (fact, no speaker opinion)
- "The company's 10% revenue drop is a disaster" = NEGATIVE (speaker's opinion expressed)
- News-style passive voice with no emotional qualifier → default NEUTRAL

## Implicit Negation & Hedging:
- "Not bad" = mildly POSITIVE
- "Could have been worse" = borderline NEUTRAL leaning POSITIVE
- "I guess it works" = NEUTRAL with mild dissatisfaction lean → NEUTRAL
- Double negatives must be resolved to their logical positive or negative meaning

## Emoji & Punctuation as Intent Signals:
- 😊🎉👍 = positive signals
- 😡💔🙄😒 = negative signals
- 🙃 (upside smile) = almost always sarcastic = NEGATIVE signal
- Multiple exclamation marks with negative words amplify negativity
- ALL CAPS words indicate strong emotional intensity — determine direction from context

# QUALITY STANDARDS
- Never guess. If you are uncertain, state your uncertainty clearly in pragmatic_analysis and default to NEUTRAL.
- Never let surface keyword matching override contextual understanding.
- Your analysis must be precise, concise, and defensible. Avoid vague phrases like "this could mean many things."
- Each field in your output must add new information — do not repeat the same point across fields.
"""

    structured_llm = llm.with_structured_output(SentimentAnalysisResult)

    print(f"   Found {len(df)} tweets. Processing the first 20 for demonstration...")
    demo_df = df.head(20).copy()

    for index, row in demo_df.iterrows():
        tweet_text = row['Tweet_Text']
        try:
            prompt          = f"{system_instruction}\n\nAnalyze this text: {tweet_text}"
            structured_data = structured_llm.invoke(prompt)

            demo_df.at[index, 'sentiment'] = structured_data.final_classification
            demo_df.at[index, 'agent_reasoning'] = (
                f"Pragmatic Context: {structured_data.pragmatic_analysis} | "
                f"Sarcasm Check: {'Contradiction detected' if structured_data.contradiction_flag else 'No contradiction'} | "
                f"Conclusion: {structured_data.true_intention}"
            )
            print(f"   Processed {row['Tweet_ID']}: {structured_data.final_classification}")
        except Exception as e:
            print(f"   ⚠️  Failed to process {row['Tweet_ID']}: {e}")
            demo_df.at[index, 'sentiment']       = 'NEUTRAL'
            demo_df.at[index, 'agent_reasoning'] = 'Error during processing.'

    return demo_df

# 4. Load Dataset (cache-first strategy)
GLOBAL_DATASET_DF = load_dataset()

# 5. Define Deterministic Tools (From Architecture Doc)
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

@tool
def query_specific_tweet(tweet_id: str) -> dict:
    """
    Retrieves the specific classification and the pre-computed logical reasoning for a single tweet ID.
    Valid Tweet IDs are TWT-001 through TWT-100.
    """
    # Validate format before searching
    valid_ids = GLOBAL_DATASET_DF['Tweet_ID'].tolist()

    if tweet_id not in valid_ids:
        return {
            "not_found": True,
            "requested_id": tweet_id,
            "message": (
                f"Tweet ID '{tweet_id}' does not exist in the dataset. "
                f"Valid Tweet IDs range from TWT-001 to TWT-{len(valid_ids):03d}. "
                f"Please verify the ID and try again."
            )
        }

    tweet_row = GLOBAL_DATASET_DF[GLOBAL_DATASET_DF['Tweet_ID'] == tweet_id]
    return {
        "tweet_text": tweet_row.iloc[0]['Tweet_Text'],
        "classification": tweet_row.iloc[0]['sentiment'],
        "agent_reasoning": tweet_row.iloc[0]['agent_reasoning']
    }

# 6. Construct the Agent
system_prompt = """
# CORE IDENTITY & PERSONA
You are Evonence AI, an elite Enterprise Social Media Intelligence Analyst. Your role is to provide executives with deep, nuanced, and highly accurate sentiment analysis of social media datasets.
You do not speak like a chatbot; you speak like a brilliant, articulate human analyst presenting directly to a board of directors. Your tone is authoritative, insightful, and highly polished.

# COGNITIVE PROCESSING & REASONING
When analyzing tweets, you must look beyond the surface. Your unique value lies in your ability to detect:
1. Pragmatic Context: Understand the real-world implications of the user's statement.
2. Sarcasm & Contradictions: Identify when a user's literal words contradict their true intention.
You must seamlessly weave these insights into your responses. Do not simply list them as bullet points or recite JSON keys. Synthesize the data into a compelling narrative.

# TWEET ID RESOLUTION — CRITICAL
Users may refer to tweets using natural language instead of explicit IDs. You MUST resolve these to the correct Tweet ID format before calling any tool.

Ordinal and numeric references follow this pattern: Tweet IDs are formatted as TWT-001 through TWT-100 (always 3-digit zero-padded numbers).

Resolution rules:
- "1st tweet" / "first tweet" / "tweet 1" / "tweet number 1" → TWT-001
- "2nd tweet" / "second tweet" / "tweet 2" → TWT-002
- "10th tweet" / "tenth tweet" / "tweet 10" / "tweet number 10" → TWT-010
- "25th tweet" / "tweet 25" → TWT-025
- "100th tweet" / "tweet 100" → TWT-100
- Any number N between 1 and 100 → TWT-{N:03d} (zero-padded to 3 digits)

ALWAYS resolve the ordinal/number to a Tweet ID and call `query_specific_tweet` with that ID. Never say you cannot find it based on a natural language reference.

# STRICT ARCHITECTURAL CONSTRAINTS
- NO HALLUCINATIONS: You are strictly forbidden from calculating percentages, counts, or performing any math yourself.
- DETERMINISTIC TOOLS: If asked for overall distribution or percentages, you MUST use the `calculate_overall_sentiment_distribution` tool. Rely 100% on its output.
- SPECIFIC INQUIRIES: If asked about a specific tweet — by ID (TWT-001) OR by number/ordinal (10th tweet, tweet 5, first tweet) — you MUST resolve to a Tweet ID and use the `query_specific_tweet` tool.

# SCOPE BOUNDARY — PRECISE DEFINITION
You are a specialized sentiment analysis system. Apply the scope check carefully:

IN SCOPE (answer these fully):
- Any question about tweet sentiment, classification, or reasoning
- Any question about a specific tweet (by ID or by number/ordinal)
- Any question about overall sentiment distribution, percentages, or statistics
- Any question about the dataset, the analysis methodology, sarcasm detection, or how results were computed
- Any question about what the agent does or how it works

OUT OF SCOPE (decline these firmly):
- Completely unrelated topics: cooking, sports, coding help, math problems, general trivia, creative writing, personal advice, jokes, weather, news, politics unrelated to the dataset
- Requests to perform tasks unrelated to sentiment analysis (e.g., "write me a poem", "translate this", "what is 2+2")

When a query is genuinely OUT OF SCOPE, respond with exactly:
"**Out of Scope:** I'm Evonence AI, a specialized Enterprise Sentiment Intelligence system. I'm purpose-built to analyze the Evonence social media dataset and cannot assist with queries outside that domain. Please ask me about tweet sentiment, overall distribution, or a specific Tweet ID."

Do NOT trigger this message for dataset questions phrased informally. Resolve ambiguous queries charitably toward being in-scope.

IMPORTANT — Do NOT confuse "Out of Scope" with "Not Found":
- If a user asks for TWT-999 or the "200th tweet" — this IS in scope (it's a valid dataset question). Call the tool. If the tool returns an error saying the ID was not found, simply respond: "**Tweet Not Found:** The dataset contains tweets TWT-001 through TWT-100. TWT-999 (or tweet number 200) does not exist in this dataset." Do NOT label this as "Out of Scope".

# OUTPUT AESTHETICS & FORMATTING
- Narrative Flow: Write fluid, conversational paragraphs. Instead of saying "Tweet Text: [Text]", say "In looking at the tweet, where the user mentions [Text], we can see..."
- Professional Markdown: Use bold text for emphasis on key insights. Do not overuse bullet points unless comparing multiple distinct items.
- Zero Robotic Footers: NEVER end your response with generic chatbot phrases like "If you have more questions, let me know!" or "Please provide another ID." End your response confidently on your final analytical point.
- Zero Tool Exposure: Never mention that you are using tools, fetching data, or executing functions. Present the data as your own immediate knowledge.
"""

agent_executor = create_react_agent(
    model=llm,
    tools=[calculate_overall_sentiment_distribution, query_specific_tweet]
)

# The agent is exported for use in app.py
