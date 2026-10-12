# **Architecting an Enterprise-Grade Social Media Sentiment Analysis Agent on Google Cloud**

## **1\. Introduction to Agentic Data Processing in Enterprise Environments**

The deployment of Large Language Models (LLMs) for macroeconomic or macro-social dataset analysis presents a fundamental computational paradox. These models demonstrate unparalleled capabilities in nuanced semantic extraction, syntactic parsing, and pragmatic interpretation. However, because they function as auto-regressive next-token predictors operating on probabilistic distributions rather than deterministic logic gates, they suffer from inherent deterministic shortcomings. This vulnerability is most pronounced when executing arithmetic aggregation and strict mathematical reasoning.  
When tasked with processing a dataset of over 1,000 social media posts (e.g., a synthetic .xlsx file of tweets) to extract both individual granular classifications and dataset-wide sentiment distributions, the architectural design must rigorously decouple semantic inference from mathematical computation. In a competitive hackathon environment, such as those hosted by Google Cloud Premier Partners like Evonence, the distinction between a prototype and an enterprise-grade solution lies in this architectural decoupling. Relying on an LLM to count and calculate percentage distributions across a large context window guarantees mathematical hallucination.  
The Google Agent Development Kit (ADK) paired with Gemini 1.5 Pro and Gemini 1.5 Flash models provides a robust, state-of-the-art framework for achieving this separation of concerns1. By leveraging the ADK's extensive session management and tool-calling capabilities alongside Vertex AI's infrastructure, engineers can construct a system that handles both highly specific queries (e.g., "Analyze the sentiment of Tweet ID 452") and macro-level aggregate queries (e.g., "What is the overall sentiment distribution of the dataset?") with zero arithmetic hallucination.

## **2\. Phase 1: Architectural & Design Patterns for Agentic Data Processing**

To identify the optimal methodology for integrating a large Excel/CSV dataset with an LLM Agent, it is necessary to evaluate the three primary architectural patterns dominating the modern AI engineering landscape: In-Memory Tool Calling, Retrieval-Augmented Generation (RAG), and Batch/Map-Reduce Pipelines. Each paradigm offers distinct advantages and liabilities regarding performance, operational cost, inference latency, and vulnerability to hallucination.

### **2.1 Evaluation of Architectural Paradigms**

The requirement dictates that the system must handle individual tweet queries, provide logical explanations for classifications, and calculate macro-level percentage summaries without the LLM hallucinating the mathematics.

| Architectural Pattern | Primary Mechanism | Latency Profile | Cost Profile | Hallucination Risk for Aggregates | System Suitability |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **In-Memory Tool Calling** | Loading the dataset into a Pandas DataFrame and exposing deterministic Python functions as actionable tools to the Google ADK Agent4. | Low (Python execution is instantaneous relative to network calls). | Low (avoids tokenizing the entire dataset into the LLM context window). | Zero (Mathematics are executed strictly by the Pandas engine, bypassing the LLM entirely). | **Optimal** for aggregate mathematics and exact deterministic data retrieval from bounded datasets. |
| **Retrieval-Augmented Generation (RAG)** | Vectorizing text, storing it in a vector database, and querying via semantic similarity search utilizing embedding proximity. | Medium (Incurs embedding generation and vector retrieval overhead). | Medium (Requires vector database infrastructure and embedding API calls). | High (Similarity search is fundamentally probabilistic and routinely fails to retrieve 100% of a dataset for exhaustive statistical calculation). | **Suboptimal**. RAG is designed for semantic search over vast, unbounded unstructured corpora, not exhaustive statistical calculation. |
| **Batch / Map-Reduce Pipelines** | Utilizing Vertex AI Batch Prediction to asynchronously process all text, storing the categorized results for later deterministic aggregation6. | High (Asynchronous background processing incurs significant wait times). | Low (Batch pricing is highly efficient compared to synchronous online prediction requests). | Zero (The LLM infers individually; the Map-Reduce layer aggregates). | **Highly Recommended** as a pre-processing step prior to the real-time conversational agent loop. |

### **2.2 The Fallacy of RAG for Exhaustive Aggregation**

While Retrieval-Augmented Generation (RAG) is the prevailing standard for document question-answering, it is uniquely unsuited for comprehensive dataset aggregation. RAG relies on calculating the cosine similarity between a user's query vector and the document vectors in a latent space. If a user asks, "What percentage of all tweets are positive?", a semantic search will only retrieve tweets that are semantically similar to the phrasing of the question, rather than returning the entire dataset required for an accurate denominator. Attempting to force RAG to retrieve 1,000 discrete records simultaneously will exceed context windows, increase token costs exponentially, and ultimately force the LLM to perform arithmetic on the retrieved context—triggering mathematical hallucinations.

### **2.3 The Recommended Hybrid Architecture: Batch Processing with In-Memory Tool Calling**

The most robust and enterprise-ready architectural pattern for this specific objective is a hybrid approach. It leverages Vertex AI Batch Prediction for initial semantic classification at scale, followed by In-Memory Tool Calling within the Google ADK for the real-time conversational interface.  
The architecture operates in two distinct, decoupled phases:

#### **2.3.1 The Background Processing Phase (Vertex AI Batch Prediction)**

The raw .xlsx dataset is parsed and passed to the Gemini 1.5 Flash model via the Vertex AI Batch Prediction API8. Using the Batch API allows the system to process thousands of records asynchronously without hitting standard online rate limits. The model evaluates each tweet individually using a highly engineered prompt and outputs a JSON Lines (JSONL) file containing strict classifications and the reasoning behind them7. Because Gemini 1.5 Flash is highly optimized for speed and cost, it is the ideal candidate for this Map-Reduce operation.

#### **2.3.2 The Conversational Phase (Google ADK & Pandas)**

The pre-processed dataset, now enriched with deterministic sentiment labels and reasoning strings, is loaded into a Pandas DataFrame in the application's memory. The Google ADK Agent (powered by Gemini 1.5 Pro for higher conversational reasoning) is equipped with specific Python tools via the @tool decorator5.  
When the user queries the overall sentiment percentage, the ADK Agent recognizes the intent and triggers a tool that executes dataframe\['sentiment'\].value\_counts(normalize=True) \* 100\. The tool returns the exact, deterministically calculated mathematical values to the LLM. The LLM then receives these hard numbers in its context window and formats the final conversational response. This architecture entirely eliminates the risk of mathematical hallucination while preserving the conversational fluidity and dynamic routing capabilities of the Agent.

## **3\. Phase 2: Advanced Prompt Engineering & Reasoning Frameworks**

Top-tier artificial intelligence systems handling complex linguistic nuances—such as implicit negation, sarcasm, and factual versus opinionated statements—rely on highly structured, multi-step reasoning frameworks. Recent research in Natural Language Processing (NLP) published between 2023 and 2025 has demonstrated conclusively that standard zero-shot prompting or generic Chain-of-Thought (CoT) prompting is insufficient for handling edge cases like sarcasm11.  
Social media text is uniquely challenging due to its informal nature, heavy reliance on contextual cues, and frequent use of figurative language. The system prompt must be engineered to deconstruct these subjective linguistic patterns systematically.

### **3.1 The Linguistic Challenges of Social Media Analysis**

The analysis of synthetic or real-world social media datasets requires overcoming three primary linguistic hurdles:

> 1. **Implicit Negation and Sarcasm:** Sarcasm is fundamentally defined by emotional incongruity—a deliberate, pragmatic mismatch between the literal meaning of the words and the speaker's true intended meaning11. For example, the statement, "The stock market crashed. Relief." pairs a disastrous macroeconomic event with a positive emotional reaction. Standard lexicon-based classifiers will detect "Relief" and score the text positively, failing to recognize the implicit negation signaling dark humor or sarcasm.  
> 2. **Factual vs. Opinionated Statements:** Standard sentiment classifiers frequently trigger false positives or false negatives on neutral, factual statements containing inherently positive or negative vocabulary. For instance, the statement "The new update dropped today" contains the word "new," which lexicon models often correlate with positive sentiment. However, in an enterprise context, this is a purely neutral, objective fact. Disentangling objective reality from subjective judgment is critical for accurate brand monitoring.  
> 3. **Contextual Emojis and Slang:** Emojis and regional slang often serve as the sole indicators of true intent, acting as subtextual emotional modifiers that flip the polarity of the preceding text13. An agent must be prompted to weigh these non-standard tokens heavily during the inference phase.

### **3.2 Scientifically-Backed Prompting Strategies**

To combat these challenges, recent academic research has introduced frameworks explicitly designed to deconstruct subjective language. Two highly effective frameworks have proven to increase accuracy in these edge cases and must be integrated into the Agent's system prompt.

#### **3.2.1 SarcasmCue and the Chain of Contradiction (CoC)**

The "SarcasmCue" framework introduces a metacognitive reasoning path tailored specifically for sarcasm, recognizing that sarcasm does not conform to standard logical syllogisms (e.g., if A implies B, and B implies C, then A implies C)11. Within this framework, the "Chain of Contradiction" (CoC) paradigm forces the LLM to step outside linear logic and directly evaluate paradoxes11.  
CoC operates by forcing the LLM through a strict three-step evaluation sequence before it is permitted to generate a final classification label:

* **Step 1: Surface Sentiment Extraction.** The model identifies the literal sentiment based purely on keywords, emojis, and explicit phrasing11.  
* **Step 2: True Intention Deduction.** The model analyzes rhetorical devices, contextual absurdities, and pragmatic realities to deduce what the speaker actually means14.  
* **Step 3: Contradiction Evaluation.** The model explicitly compares the surface sentiment with the true intention. If a contradiction exists, the statement is classified based on the true intention, thereby capturing the sarcasm accurately11.

#### **3.2.2 Pragmatic Metacognitive Prompting (PMP)**

Developed to improve LLM performance on complex contextual dialogue, Pragmatic Metacognitive Prompting (PMP) forces the model to analyze the underlying pragmatics of a statement before making a classification15. PMP requires the model to analyze three distinct dimensions of the text:

* **Implicature:** What is implied beyond the literal text?15  
* **Presuppositions:** What background information or real-world facts are taken for granted by the speaker?15  
* **Speaker Intent:** What is the speaker attempting to achieve (e.g., sharing an objective fact, expressing personal frustration, mocking a corporate decision)?15

### **3.3 The Optimal System Prompt Template**

By synthesizing the Chain of Contradiction (CoC) methodology with Pragmatic Metacognitive Prompting (PMP), the following system prompt template is engineered to maximize accuracy against sarcasm and factual traps. This prompt is designed for the Vertex AI Batch Prediction pipeline to classify the raw data accurately.  
You are an expert linguistic analyst and enterprise social media sentiment classifier. Your objective is to classify the provided social media text strictly as POSITIVE, NEGATIVE, or NEUTRAL.  
You must meticulously differentiate between factual reporting (Neutral) and opinionated sentiment (Positive/Negative). Furthermore, you must aggressively detect sarcasm and implicit negation.  
To achieve this, you must execute the following Pragmatic Metacognitive and Chain of Contradiction reasoning steps internally before outputting your classification. Do not output the classification until these steps are complete.  
STEP 1: PRAGMATIC ANALYSIS

* Implicature: What is implied beyond the literal words?  
* Presuppositions: What world facts are assumed? (e.g., if a company stock crashed, it is objectively bad).  
* Fact vs. Opinion: Is the speaker stating an objective reality (e.g., "The product was released") or a subjective judgment (e.g., "The product is revolutionary")? Factual reporting must be classified as NEUTRAL, regardless of the presence of words like "new", "growth", or "drop".

STEP 2: CHAIN OF CONTRADICTION (Sarcasm Detection)

* Surface Sentiment: What is the literal sentiment of the words and emojis?  
* True Intention: Given the pragmatic analysis and real-world context, what is the speaker's actual intent?  
* Contradiction Check: Does the surface sentiment directly contradict the true intention? (e.g., "Great, another delay." \-\> Surface: Positive, True Intent: Negative due to frustration). If a contradiction exists, the True Intention overrides the Surface Sentiment.

STEP 3: FINAL CLASSIFICATION Based on the synthesized analysis above, provide the final sentiment classification.

## **4\. Phase 3: Hallucination Mitigation & Output Formatting**

Even with advanced, scientifically-backed prompt engineering, Large Language Models can exhibit non-deterministic output structures. If a model generates text such as "I believe this is Positive because...", it breaks downstream parsing systems that expect a strict categorical label. Furthermore, without rigorous constraints, conversational agents are prone to inventing external facts not present in the source material to justify their reasoning.

### **4.1 Enforcing Strict Output Schemas via Pydantic**

To mitigate these risks and enforce strict output schemas, the system must leverage Structured Outputs. Google's Generative AI SDK (google-genai) and the Gemini 1.5 models natively support JSON Schema and Pydantic integration16.  
By defining a Pydantic model and passing it to the response\_schema parameter within the API call, while simultaneously setting the response\_mime\_type to application/json, the model is forced at the deepest API level to adhere strictly to the defined data structure18. If the model attempts to output plain text or deviate from the keys defined in the schema, the API will constrain the output to ensure 100% valid JSON generation.  
To prevent the Agent from inventing external facts, the schema design itself is used as a psychological constraint mechanism. Because LLMs generate tokens sequentially, forcing the model to write out its "Chain of Contradiction" analysis and "Pragmatic Analysis" into specific JSON fields *before* it generates the final classification field acts as an internal guardrail. This prevents the model from hastily jumping to a conclusion and post-hoc rationalizing a hallucination. It forces the model to ground its reasoning strictly in the context of the provided text.

### **4.2 State Management and Context Preservation in Google ADK**

The Google Agent Development Kit (ADK) manages conversation context through its robust SessionService architecture20. To prevent context window bloat—which degrades model attention and increases the likelihood of hallucination during multi-turn conversations—state management must be strictly controlled.  
The ADK utilizes a Session object containing events (the full chat history array) and state (a key-value store acting as the agent's scratchpad)20. The state can be dynamically updated using CallbackContext or ToolContext during execution5.  
For the hackathon implementation, utilizing the InMemorySessionService is optimal for rapid prototyping and local debugging via the adk web interface20. However, the architecture must ensure that the raw text of the 1,000+ tweets is *never* injected directly into the session state. Instead, the pre-processed DataFrame resides in the Python application memory, and the Agent only accesses discrete slices of this data when a specific tool is called5. This retrieves only the necessary subset of information, keeping the context window pristine and entirely eliminating the risk of the model synthesizing facts from overlapping rows of data.

## **5\. System Implementation and Workflow Design**

The following technical implementation guide details the specific Python pseudo-code required to build this hybrid architecture using the Google ADK, Gemini 1.5 Pro/Flash, and the google-genai SDK. This code demonstrates the extraction of the sentiment percentage across the entire dataset without requiring the LLM to perform arithmetic.

### **5.1 Stage 1: Batch Pre-Processing via Structured Outputs**

This script simulates the pre-processing pipeline. It utilizes the Gemini 1.5 Flash model and a Pydantic schema to enforce the Chain of Contradiction reasoning steps, ensuring the output is perfectly formatted for downstream ingestion.

Python  
import pandas as pd  
from google import genai  
from pydantic import BaseModel, Field  
from typing import Literal

\# Define the Strict Output Schema enforcing Chain of Contradiction and PMP  
class SentimentAnalysisResult(BaseModel):  
    pragmatic\_analysis: str \= Field(description="Analysis of fact vs opinion and implicit context.")  
    surface\_sentiment: str \= Field(description="The literal sentiment of the words.")  
    true\_intention: str \= Field(description="The actual intent of the speaker.")  
    contradiction\_flag: bool \= Field(description="True if surface sentiment contradicts true intention.")  
    final\_classification: Literal\["POSITIVE", "NEGATIVE", "NEUTRAL"\] \= Field(description="The final strict categorical classification.")

def preprocess\_social\_dataset(file\_path: str, api\_key: str) \-\> pd.DataFrame:  
    """  
    Simulates a batch processing pipeline to classify all text deterministically.  
    In a production environment, this function would wrap the Vertex AI BatchPredictionJob API.  
    """  
    df \= pd.read\_excel(file\_path)  
    client \= genai.Client(api\_key=api\_key)  
      
    \# Pre-allocate columns for the results  
    df\['sentiment'\] \= 'NEUTRAL'  
    df\['agent\_reasoning'\] \= ''  
      
    system\_instruction \= """  
    You are an expert linguistic analyst. Classify text strictly as POSITIVE, NEGATIVE, or NEUTRAL.  
    Differentiate factual reporting (Neutral) from opinionated sentiment.   
    Detect sarcasm by checking for contradictions between literal words and true intent.  
    Follow the JSON schema strictly.  
    """  
      
    \# Iterate through the dataset (For production, batch asynchronous requests are required)  
    for index, row in df.iterrows():  
        tweet\_text \= row\['Tweet\_Text'\]  
          
        response \= client.models.generate\_content(  
            model='gemini-1.5-flash',  
            contents=f"Analyze this text: {tweet\_text}",  
            config={  
                'system\_instruction': system\_instruction,  
                'response\_mime\_type': 'application/json',  
                'response\_schema': SentimentAnalysisResult,  
                'temperature': 0.1 \# Minimal temperature enforces deterministic, grounded classification  
            }  
        )  
          
        \# Parse the structured Pydantic output natively via the SDK  
        structured\_data \= response.parsed  
        df.at\[index, 'sentiment'\] \= structured\_data.final\_classification  
          
        \# Concatenate reasoning steps to provide a logical explanation for the user later  
        reasoning\_string \= (  
            f"Pragmatic Context: {structured\_data.pragmatic\_analysis} | "  
            f"Sarcasm Check: {'Contradiction detected' if structured\_data.contradiction\_flag else 'No contradiction'} | "  
            f"Conclusion: {structured\_data.true\_intention}"  
        )  
        df.at\[index, 'agent\_reasoning'\] \= reasoning\_string  
          
    return df

### **5.2 Stage 2: Google ADK Agent Orchestration and Tool Calling**

Once the data is pre-processed and enriched, it is passed into the ADK Agent's environment. The LLM is provided with Python tools that perform the deterministic mathematics and data retrieval.

Python  
import os  
from google.adk.agents import LlmAgent  
from google.adk.tools import ToolContext, tool  
from google.adk.sessions import InMemorySessionService  
from google.adk.runners import Runner

\# Ingest the pre-processed dataset into memory  
\# In a robust production environment, this would interface with BigQuery or Cloud SQL  
GLOBAL\_DATASET\_DF \= preprocess\_social\_dataset("synthetic\_tweets.xlsx", os.getenv("GOOGLE\_API\_KEY"))

@tool  
def calculate\_overall\_sentiment\_distribution(tool\_context: ToolContext) \-\> dict:  
    """  
    Calculates the macro-level percentage distribution of sentiments across the entire dataset.  
    The agent MUST use this tool to answer questions about overall percentages or statistics.  
    """  
    total\_records \= len(GLOBAL\_DATASET\_DF)  
    if total\_records \== 0:  
        return {"error": "Dataset is currently empty."}  
          
    \# The Pandas engine handles the mathematics deterministically, preventing LLM arithmetic hallucination  
    distribution \= GLOBAL\_DATASET\_DF\['sentiment'\].value\_counts(normalize=True) \* 100  
      
    return {  
        "total\_tweets\_analyzed": total\_records,  
        "positive\_percentage": round(distribution.get('POSITIVE', 0), 2),  
        "negative\_percentage": round(distribution.get('NEGATIVE', 0), 2),  
        "neutral\_percentage": round(distribution.get('NEUTRAL', 0), 2)  
    }

@tool  
def query\_specific\_tweet(tool\_context: ToolContext, tweet\_id: str) \-\> dict:  
    """  
    Retrieves the specific classification and the pre-computed logical reasoning for a single tweet ID.  
    """  
    tweet\_row \= GLOBAL\_DATASET\_DF\[GLOBAL\_DATASET\_DF\['Tweet\_ID'\] \== tweet\_id\]  
      
    if tweet\_row.empty:  
        return {"error": f"Tweet ID {tweet\_id} could not be located in the dataset."}  
          
    return {  
        "tweet\_text": tweet\_row.iloc\[0\]\['Tweet\_Text'\],  
        "classification": tweet\_row.iloc\[0\]\['sentiment'\],  
        "agent\_reasoning": tweet\_row.iloc\[0\]\['agent\_reasoning'\]  
    }

\# Construct the ADK Agent  
sentiment\_assistant \= LlmAgent(  
    name="SocialSentimentAssistant",  
    model="gemini-1.5-pro", \# Utilizing the Pro model for superior conversational routing  
    instruction="""  
    You are an enterprise AI Assistant focused on Social Media Sentiment Analysis.  
    You interact conversationally with users to explain the sentiment of a large dataset.  
      
    CRITICAL ARCHITECTURAL CONSTRAINTS:  
    1\. NEVER attempt to calculate percentages, counts, or math yourself. You MUST use the \`calculate\_overall\_sentiment\_distribution\` tool to obtain these metrics.  
    2\. When asked about a specific tweet, use the \`query\_specific\_tweet\` tool.  
    3\. Always explain the logic supporting a classification clearly to the user, using the reasoning data provided by the tool. Do not invent external facts.  
    """,  
    tools=\[calculate\_overall\_sentiment\_distribution, query\_specific\_tweet\]  
)

\# Initialize Session Management and Runner  
session\_service \= InMemorySessionService()

\# The Runner orchestrates the connection between the Agent, the Tools, and the Session State  
runner \= Runner(  
    app\_name="EvonenceHackathonAgent",  
    agent=sentiment\_assistant,  
    session\_service=session\_service  
)

By explicitly commanding the agent to use the deterministic tool (calculate\_overall\_sentiment\_distribution), the system offloads the Map-Reduce aggregation entirely to the Pandas engine. The Agent receives a highly structured JSON response from the Python function and is only responsible for its core competency: translating those hard numbers into fluid, natural language for the end user.

## **6\. Evaluation: Markers of a "Hackathon Winning" Implementation**

In a competitive engineering environment evaluated by Google Cloud Premier Partners, judges look for implementations that transcend standard wrapper applications. A winning implementation demonstrates a profound understanding of enterprise-grade constraints, infrastructure scalability, and Google Cloud best practices.

> 1. **Mathematical Determinism via Tool Calling:** The most common failure mode in generative AI hackathons is relying on the model to perform aggregate math over large context windows. Demonstrating that the system securely isolates mathematical computation in a deterministic Python tool while using the LLM exclusively for routing and semantic formatting proves a deep understanding of LLM limitations and systems architecture.  
> 2. **Implementation of Advanced Academic Frameworks:** Generic prompts (e.g., "Analyze the sentiment of this text") perform poorly in production. Citing and implementing recent NLP research—specifically the "Chain of Contradiction" and "Pragmatic Metacognitive Prompting" frameworks—to handle edge cases like sarcasm and factual reporting demonstrates advanced AI engineering maturity11.  
> 3. **Strict Output Enforcement:** Utilizing Pydantic schemas in conjunction with the google-genai SDK's structured output capabilities to force the model to output reasoning steps prior to final classification guarantees schema adherence16. This design choice mitigates hallucinations and ensures the pipeline will not break downstream database ingestion processes.  
> 4. **Robust ADK State Management:** Moving beyond stateless, isolated API calls by implementing the ADK's SessionService shows readiness for production deployment20. Utilizing ToolContext to isolate data operations from the context window prevents token overflow and latency degradation as the conversation history grows5.  
> 5. **Interactive Developer UX:** Utilizing the adk web command line interface to spin up a local development UI for the judges demonstrates an understanding of the Google Agent Development Kit's built-in debugging and visualization capabilities, making the final presentation highly interactive and visually compelling1.

## **7\. Essential Reference Architectures and Literature**

To fulfill the requirements of delivering scientifically-backed methodologies and Google Cloud best practices, the following resources, literature, and reference architectures form the foundation of this report's recommendations.

| Resource Category | Description and Link / Reference |
| :---- | :---- |
| **NLP Research (Sarcasm & Pragmatics)** | *Is Sarcasm Detection a Step-by-Step Reasoning Process in Large Language Models?* (Yao et al., 2024). Details the SarcasmCue and Chain of Contradiction (CoC) frameworks. \[Link: https\://arxiv.org/abs/2407.12725\] |
| **NLP Research (Contextual Prompts)** | *Pragmatic Metacognitive Prompting Improves LLM Performance on Sarcasm Detection.* (Lee et al., 2024). Details the PMP framework for isolating implicature and presuppositions. \[Link: https\://arxiv.org/abs/2412.04509\] |
| **Google Cloud Architecture** | *Generative AI on Vertex AI: Control Generated Output.* Official documentation detailing how to enforce strict JSON schemas and Pydantic models using the google-genai SDK. \[Link: https\://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/control-generated-output\] |
| **Google ADK Documentation** | *Google Agent Development Kit (ADK) Repository.* Open-source documentation for configuring Agents, ToolContext, and Session Services. \[Link: https\://github.com/google/adk-python\] |
| **Vertex AI Batch Processing** | *Get Batch Predictions for Gemini.* Reference architecture for executing large-scale, asynchronous model inferences via JSONL to reduce cost and latency constraints. \[Link: https\://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/batch-prediction-api\] |

By synthesizing these advanced prompting frameworks with deterministic tool calling and structured output enforcement, the resulting social media sentiment agent will operate with unparalleled accuracy, speed, and mathematical reliability.

#### **Works cited**

> 1. GitHub \- google/adk-python: An open-source, code-first Python, [https\://github.com/google/adk-python](https://github.com/google/adk-python)  
> 2. Agent Development Kit | Gemini Enterprise Agent Platform, [https\://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)  
> 3. Python \- Agent Development Kit (ADK), [https\://adk.dev/get-started/python/](https://adk.dev/get-started/python/)  
> 4. I Built an AI-Powered GitHub Bot Using Google's Agent ... \- Medium, [https\://medium.com/@ashutoshkumars1ngh/i-built-an-ai-powered-github-bot-using-googles-agent-development-kit-here-s-how-you-can-too-2bc623b790c4](https://medium.com/@ashutoshkumars1ngh/i-built-an-ai-powered-github-bot-using-googles-agent-development-kit-here-s-how-you-can-too-2bc623b790c4)  
> 5. Agent context \- Agent Development Kit (ADK), [https\://adk.dev/context/](https://adk.dev/context/)  
> 6. All Vertex AI code samples | Google Cloud Documentation, [https\://cloud.google.com/vertex-ai/docs/samples](https://cloud.google.com/vertex-ai/docs/samples)  
> 7. Batch API | Gemini API \- Google AI for Developers, [https\://ai.google.dev/gemini-api/docs/batch-api](https://ai.google.dev/gemini-api/docs/batch-api)  
> 8. Get batch predictions for Gemini \- Vertex AI returns unsupported, [https\://stackoverflow.com/questions/78582566/get-batch-predictions-for-gemini-vertex-ai-returns-unsupported](https://stackoverflow.com/questions/78582566/get-batch-predictions-for-gemini-vertex-ai-returns-unsupported)  
> 9. Get batch predictions for Gemini | Gemini Enterprise Agent Platform, [https\://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/batch-prediction-api](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/batch-prediction-api)  
> 10. How Do Agents Access State Variables? Understanding State Value, [https\://github.com/google/adk-python/discussions/2489](https://github.com/google/adk-python/discussions/2489)  
> 11. arXiv:2407.12725v2 \[cs.CL\] 24 Aug 2024, [https\://arxiv.org/pdf/2407.12725](https://arxiv.org/pdf/2407.12725)  
> 12. Detecting Emotional Incongruity of Sarcasm by Commonsense, [https\://www\.researchgate.net/publication/387141255\_Detecting\_Emotional\_Incongruity\_of\_Sarcasm\_by\_Commonsense\_Reasoning](https://www.researchgate.net/publication/387141255_Detecting_Emotional_Incongruity_of_Sarcasm_by_Commonsense_Reasoning)  
> 13. A systematic review of sarcasm recognition — multimodal fusion, [https\://arxiv.org/html/2509.04605v1](https://arxiv.org/html/2509.04605v1)  
> 14. Is Sarcasm Detection A Step-by-Step Reasoning Process in ... \- arXiv, [https\://arxiv.org/html/2407.12725v1](https://arxiv.org/html/2407.12725v1)  
> 15. arXiv:2412.04509v1 \[cs.CL\] 4 Dec 2024, [https\://arxiv.org/pdf/2412.04509](https://arxiv.org/pdf/2412.04509)  
> 16. Structured outputs \- Interactions API \- Google AI for Developers, [https\://ai.google.dev/gemini-api/docs/structured-output](https://ai.google.dev/gemini-api/docs/structured-output)  
> 17. Improving Structured Outputs in the Gemini API, [https\://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/)  
> 18. Structured output | Gemini Enterprise Agent Platform, [https\://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/control-generated-output](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/control-generated-output)  
> 19. Structured output for open models | Gemini Enterprise Agent Platform, [https\://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/capabilities/structured-output](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/capabilities/structured-output)  
> 20. Google ADK Session and State Management: Understanding, [https\://medium.com/google-cloud/google-adk-session-and-state-management-understanding-sessions-and-state-a5e05b62f1f1](https://medium.com/google-cloud/google-adk-session-and-state-management-understanding-sessions-and-state-a5e05b62f1f1)  
> 21. Google ADK Masterclass Part 5: Session and Memory Management, [https\://saptak.in/writing/2025/05/10/google-adk-masterclass-part5](https://saptak.in/writing/2025/05/10/google-adk-masterclass-part5)  
> 22. Debug and Inject Data/Session to the Context in ADK \- Medium, [https\://medium.com/@juanc.olamendy/debug-and-inject-data-session-to-the-context-in-adk-128396ced491](https://medium.com/@juanc.olamendy/debug-and-inject-data-session-to-the-context-in-adk-128396ced491)  
> 23. Remember this: Agent state and memory with ADK \- Google Cloud, [https\://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk](https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk)  
> 24. Agent Development Kit Web (adk web) is the built-in developer UI, [https\://github.com/google/adk-web](https://github.com/google/adk-web)

## **6. Phase 4: Enterprise Security & Governance Layer**

The deployment of an LLM agent in an enterprise environment necessitates strict security boundaries to protect the underlying infrastructure from malicious exploitation, denial-of-service, and prompt injection attacks.

### **6.1 In-Memory Rate Limiting & DoS Prevention**
To prevent API abuse and cost overruns from rapid, repetitive LLM invocations, a lightweight, in-memory sliding window rate limiter has been implemented at the routing layer (`app.py`). It strictly limits incoming requests to 5 per minute per IP address, gracefully rejecting excess requests with HTTP 429 status codes. This mitigates automated bot traffic and Denial-of-Wallet attacks.

### **6.2 Input Sanitization & Payload Constraints**
To defend against complex prompt injection or buffer overflow vectors, strict input validation constraints exist at the API ingress point. All user input is rigorously truncated and validated (max 500 characters) before being passed to the Bedrock Converse API, ensuring optimal context window utilization and reducing injection surfaces.

### **6.3 Hardened HTTP Headers & CSP**
The Flask application is fortified with robust HTTP security headers, including:
- **X-Content-Type-Options: nosniff**
- **X-Frame-Options: DENY** (Prevents Clickjacking)
- **X-XSS-Protection: 1; mode=block** (Mitigates Cross-Site Scripting)
- **Content-Security-Policy (CSP):** Strictly whitelists execution of inline scripts and external CDNs, fully neutralizing malicious injection payloads on the frontend interface.