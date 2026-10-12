import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

# 1. Securely load credentials from .env file
load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not access_key or not secret_key:
    raise ValueError("AWS credentials are missing from the .env file!")

# 2. Initialize the Bedrock client
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

from pydantic import BaseModel, Field
from typing import Literal

# 3. Define the Strict Output Schema enforcing Chain of Contradiction (From Architecture Doc)
class SentimentAnalysisResult(BaseModel):
    pragmatic_analysis: str = Field(description="Analysis of fact vs opinion and implicit context.")
    surface_sentiment: str = Field(description="The literal sentiment of the words.")
    true_intention: str = Field(description="The actual intent of the speaker.")
    contradiction_flag: bool = Field(description="True if surface sentiment contradicts true intention.")
    final_classification: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(description="The final strict categorical classification.")
