# Prompt Engineering Guide — Evonence AI Project
### Interview Prep: Simple, Sorted, Speakable

---

## 1. What is Prompt Engineering?

Prompt engineering is the **art of writing instructions for an AI** to get the best possible output.

> Simple answer for interviewer:
> *"Prompt engineering is writing clear, structured instructions that tell the AI what role to play, how to think, what rules to follow, and what format to output — instead of just asking a plain question."*

---

## 2. Types of Prompt Engineering (The Main Ones)

| Type | What It Means | Simple Example |
| :--- | :--- | :--- |
| **Role / Persona Prompting** | Tell the AI who to act as | *"You are an expert linguist with 15 years of experience..."* |
| **Instruction Prompting** | Give the AI clear step-by-step rules | *"First do X, then do Y, then classify..."* |
| **Chain of Thought (CoT)** | Force the AI to reason step-by-step before answering | *"Think through surface sentiment, then true intention, then decide"* |
| **Constraint Prompting** | Tell the AI what it is NOT allowed to do | *"NEVER calculate math yourself. NEVER say robotic footers."* |
| **Few-Shot Prompting** | Give examples inside the prompt | *"'Not bad' = POSITIVE. 'I guess it works' = NEUTRAL"* |
| **Structured Output Prompting** | Force the AI to return a specific format | Using Pydantic schema with `with_structured_output()` |
| **Scope / Boundary Prompting** | Define what the AI should and should not answer | *"OUT OF SCOPE: cooking, sports, math. IN SCOPE: dataset questions."* |
| **Negative Prompting** | Explicitly list things the AI must avoid | *"Never mention tools. Never end with 'Let me know!'"* |

---

## 3. What Prompts We Used in the Project

Our project has **two different prompts** for two different purposes.

---

### Prompt 1 — `system_instruction` (Used During Data Pre-processing)

**Where:** `main.py` and `batch_enrich_dataset.py` — used when Amazon Nova Pro classifies each tweet.

**Purpose:** Tell Nova Pro *how to think* before classifying a tweet as POSITIVE / NEGATIVE / NEUTRAL.

**Prompt Engineering Types Used:**

| Technique | Where We Used It |
| :--- | :--- |
| **Role / Persona Prompting** | *"You are Dr. Evonence, a world-class computational linguist..."* — gives the AI an expert identity |
| **Chain of Thought (CoT)** | 3-step methodology: STEP 1 (Context) → STEP 2 (Surface vs True Intent) → STEP 3 (Decision) |
| **Instruction Prompting** | Numbered rules in strict priority order for the final classification decision |
| **Few-Shot Prompting** | Inline examples: *"Oh great, flight cancelled AGAIN. Fantastic!" = NEGATIVE (sarcasm)* |
| **Structured Output Prompting** | Pydantic `SentimentAnalysisResult` forces output into exact fields (pragmatic_analysis, surface_sentiment, etc.) |
| **Constraint Prompting** | *"Never guess. Never let surface keyword matching override context."* |

**Why we used Chain of Thought here:**
> Standard AI classification looks at words and decides fast. Chain of Thought forces the model to first understand context, then compare surface vs real meaning, then check for contradiction — which is how sarcasm is caught. Without CoT, `"Great, another delay!"` would be classified POSITIVE.

---

### Prompt 2 — `system_prompt` (Used for the Conversational Agent)

**Where:** `main.py` — passed to the LangGraph ReAct agent as the system message on every user chat.

**Purpose:** Tell the agent *who it is*, *what it can do*, *what it cannot do*, and *how to respond*.

**Prompt Engineering Types Used:**

| Technique | Where We Used It |
| :--- | :--- |
| **Role / Persona Prompting** | *"You are Evonence AI, an elite Enterprise Social Media Intelligence Analyst... speak like a brilliant human analyst presenting to a board."* |
| **Constraint Prompting (Anti-hallucination)** | *"STRICTLY FORBIDDEN from calculating percentages yourself. Use the tool."* |
| **Instruction Prompting (Tool Routing)** | *"If asked for distribution → use `calculate_overall_sentiment_distribution`. If asked for a tweet → use `query_specific_tweet`."* |
| **Scope / Boundary Prompting** | Explicit IN SCOPE list and OUT OF SCOPE list so the agent knows what to answer and what to decline |
| **Negative Prompting** | *"NEVER mention you are using tools. NEVER end with 'Let me know if you need help!'"* |
| **Few-Shot Prompting** | Ordinal resolution examples: *"10th tweet → TWT-010, first tweet → TWT-001"* |

**Why we used Constraint Prompting for math:**
> LLMs hallucinate numbers. If you ask "what percentage are positive?" the model might say "around 45%" from memory — which is wrong. The constraint *"NEVER do math yourself, call the tool"* forces the agent to always use the Pandas calculation, guaranteeing 100% accurate numbers.

---

## 4. How We Built the Project Using AI (Antigravity)

The interviewer may ask: *"You used AI to build this — so what prompts did you write to the AI assistant?"*

Here is how to answer simply:

**Types of prompts we gave to Antigravity (the AI coding assistant):**

| What We Asked | Prompt Type |
| :--- | :--- |
| *"Inspect my codebase and explain the architecture"* | **Instruction Prompt** |
| *"Create a README with full technical explanation"* | **Instruction + Output Format Prompt** |
| *"You are a senior technical interviewer — give me 50 Q&As"* | **Role + Instruction Prompt** |
| *"Fix the bug where 'Network error' appears instead of agent response"* | **Debugging Instruction Prompt** |
| *"The prompt is too shallow, make it enterprise-grade with Chain of Contradiction methodology"* | **Constraint + Quality Improvement Prompt** |
| *"Don't change anything else, just update this one section"* | **Scoped Constraint Prompt** |

**Key point for interview:**
> *"We didn't just ask the AI to 'write code'. We gave it precise role instructions, constraints, output format requirements, and scope boundaries — just like we do when prompting Amazon Nova Pro inside our own agent. The same prompt engineering principles apply."*

---

## 5. The Chain of Contradiction — Our Core Technique (Explain Simply)

This is the most important technique to explain. Here's how to say it simply:

**What sarcasm is:**
> Words say one thing, person means the opposite.
> `"Oh fantastic, the server crashed again!"` — words are positive, meaning is negative.

**What Chain of Contradiction does:**
> It forces the AI to check: *"Does what the words say match what the person actually means?"*
> If NO → contradiction detected → classify by TRUE MEANING, not surface words.

**3-step flow (speak this in interview):**
```
Step 1: What is the context? (Who is speaking? What situation?)
Step 2: What do the words literally say? vs What does the person actually mean?
Step 3: Is there a contradiction? → If yes, flip the label. If no, classify normally.
```

**Simple sarcasm example:**
```
Tweet: "Great, another 3-hour wait at customer support. Love this service!"

Step 1: Customer complaining about wait time
Step 2: Surface = POSITIVE ("Great", "Love")
         True intent = NEGATIVE (3-hour wait = frustration)
Step 3: CONTRADICTION detected → Final label = NEGATIVE ✅
```

Without Chain of Contradiction, a basic model would say POSITIVE (because of "Great" and "Love"). Ours says NEGATIVE. That's the value.

---

## 6. Quick One-Line Answers for Interview

| Question | Answer |
| :--- | :--- |
| What is prompt engineering? | Writing structured instructions to make an AI behave precisely and reliably. |
| What type of prompting did you use? | Role prompting, Chain of Thought, Constraint prompting, Structured Output, and Scope Boundary prompting. |
| Why Chain of Thought? | To force step-by-step reasoning before classification, which is required to detect sarcasm. |
| Why did you add constraints? | To prevent the AI from hallucinating numbers or going out of scope. |
| What is a system prompt? | A hidden instruction sent to the AI before every conversation that defines its role, rules, and behavior. |
| Did you use few-shot prompting? | Yes — we gave inline examples like `"Not bad" = POSITIVE` and `"10th tweet → TWT-010"` inside the prompt. |
| What is structured output prompting? | Forcing the AI to return data in a fixed schema (using Pydantic) instead of free text, so every field is always filled correctly. |
| How does Antigravity help with prompt engineering? | Antigravity is an AI coding assistant — we used the same prompt engineering principles to instruct it: giving it roles, constraints, output formats, and scope boundaries. |
