"""
batch_enrich_dataset.py
=======================
ONE-TIME OFFLINE SCRIPT — Run this BEFORE your demo to pre-compute sentiment
for all 100 tweets using Amazon Nova Pro and save to a CSV cache file.

At demo time, main.py loads this CSV instantly (< 0.5s) instead of calling
the Bedrock API 20+ times (which takes ~30-60 seconds).

Usage:
    cd code
    python batch_enrich_dataset.py
"""

import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_aws import ChatBedrock

# ── 1. Load credentials ────────────────────────────────────────────────────────
load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not access_key or not secret_key:
    raise ValueError("AWS credentials are missing from the .env file!")

# ── 2. Bedrock client & model ──────────────────────────────────────────────────
boto3_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key
)

llm = ChatBedrock(
    client=boto3_client,
    model_id='amazon.nova-pro-v1:0',
    model_kwargs=dict(temperature=0.1)
)

# ── 3. Pydantic schema (same as main.py) ───────────────────────────────────────
class SentimentAnalysisResult(BaseModel):
    pragmatic_analysis: str = Field(description="Analysis of fact vs opinion and implicit context.")
    surface_sentiment: str = Field(description="The literal sentiment of the words.")
    true_intention: str = Field(description="The actual intent of the speaker.")
    contradiction_flag: bool = Field(description="True if surface sentiment contradicts true intention.")
    final_classification: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="The final strict categorical classification."
    )

# ── 4. File paths ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILE   = os.path.join(BASE_DIR, "..", "Synthetic Data - Social Media Tweets.xlsx")
CACHE_FILE   = os.path.join(BASE_DIR, "enriched_tweets.csv")

# ── 5. Load Excel ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Evonence AI — Batch Enrichment Script")
print("  Model: Amazon Nova Pro (amazon.nova-pro-v1:0)")
print("=" * 60)

try:
    df = pd.read_excel(EXCEL_FILE)
    print(f"\n✅ Loaded Excel file: {len(df)} tweets found.")
except Exception as e:
    print(f"\n❌ Could not load Excel file: {e}")
    exit(1)

# ── 6. Setup ───────────────────────────────────────────────────────────────────
df['sentiment']       = 'NEUTRAL'
df['agent_reasoning'] = ''

structured_llm = llm.with_structured_output(SentimentAnalysisResult)

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

# ── 7. Process ALL 100 tweets ──────────────────────────────────────────────────
total    = len(df)
success  = 0
failed   = 0

print(f"\n🚀 Starting batch processing of all {total} tweets...\n")

for index, row in df.iterrows():
    tweet_id   = row['Tweet_ID']
    tweet_text = row['Tweet_Text']

    try:
        prompt          = f"{system_instruction}\n\nAnalyze this text: {tweet_text}"
        structured_data = structured_llm.invoke(prompt)

        df.at[index, 'sentiment'] = structured_data.final_classification
        df.at[index, 'agent_reasoning'] = (
            f"Pragmatic Context: {structured_data.pragmatic_analysis} | "
            f"Sarcasm Check: {'Contradiction detected' if structured_data.contradiction_flag else 'No contradiction'} | "
            f"Conclusion: {structured_data.true_intention}"
        )
        success += 1
        print(f"  [{index+1:>3}/{total}] ✅ {tweet_id}: {structured_data.final_classification}")

    except Exception as e:
        df.at[index, 'sentiment']       = 'NEUTRAL'
        df.at[index, 'agent_reasoning'] = f"Batch processing error: {str(e)}"
        failed += 1
        print(f"  [{index+1:>3}/{total}] ⚠️  {tweet_id}: FAILED — {e}")

# ── 8. Save to CSV cache ───────────────────────────────────────────────────────
df.to_csv(CACHE_FILE, index=False)

print("\n" + "=" * 60)
print(f"  ✅ Done! {success} succeeded, {failed} failed.")
print(f"  💾 Cache saved to: enriched_tweets.csv")
print(f"  ⚡ Next startup will load from cache instantly.")
print("=" * 60)
