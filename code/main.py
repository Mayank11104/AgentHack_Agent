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

def preprocess_social_dataset(file_path: str) -> pd.DataFrame:
    print("Pre-processing dataset using Claude 4.6 Sonnet structured outputs...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Warning: Could not load excel file: {e}")
        return pd.DataFrame(columns=["Tweet_ID", "Tweet_Text", "sentiment", "agent_reasoning"])

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

# 4. Run Pre-processing Phase
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "Synthetic Data - Social Media Tweets.xlsx")
GLOBAL_DATASET_DF = preprocess_social_dataset(DATA_FILE)

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
    """
    tweet_row = GLOBAL_DATASET_DF[GLOBAL_DATASET_DF['Tweet_ID'] == tweet_id]
    
    if tweet_row.empty:
        return {"error": f"Tweet ID {tweet_id} could not be located in the dataset."}
        
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

# STRICT ARCHITECTURAL CONSTRAINTS
- NO HALLUCINATIONS: You are strictly forbidden from calculating percentages, counts, or performing any math yourself.
- DETERMINISTIC TOOLS: If asked for overall distribution or percentages, you MUST use the `calculate_overall_sentiment_distribution` tool. Rely 100% on its output.
- SPECIFIC INQUIRIES: If asked about a specific tweet ID (e.g., TWT-001), you MUST use the `query_specific_tweet` tool.

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
