import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_aws import ChatBedrock

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
