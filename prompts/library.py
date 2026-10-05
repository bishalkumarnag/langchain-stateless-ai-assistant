"""
library.py - Predefined LangChain Prompt Templates

Standard ChatPromptTemplate instances for various assistant tasks:
- General Conversational Assistant
- Code Analysis & Explainer
- Structured Entity Extraction
- Text Summarization
"""

from langchain_core.prompts import ChatPromptTemplate

# 1. Default Assistant System Persona
DEFAULT_SYSTEM_PROMPT = """You are a helpful, knowledgeable, and precise AI assistant.
Key instructions:
1. Provide accurate, clear, and concise responses.
2. Note that each interaction is standalone (stateless). Do not assume past context unless provided in the current prompt.
3. Maintain a professional, polite, and encouraging tone."""

# 2. General Conversational Chat Prompt
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DEFAULT_SYSTEM_PROMPT),
    ("human", "{input}"),
])

# 3. Code Explainer Prompt
CODE_EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software engineer and code reviewer."),
    (
        "human",
        """Please analyze and explain the following {language} code snippet:

```{language}
{code}
```

Format your response with:
1. High-Level Summary
2. Line-by-Line Breakdown
3. Time & Space Complexity
4. Potential Edge Cases & Improvements""",
    ),
])

# 4. Text Summarization Prompt
TEXT_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert editor who produces concise summaries."),
    (
        "human",
        """Summarize the following passage in {target_length} format with focus on {focus_area}:

\"\"\"
{text}
\"\"\"

Summary:""",
    ),
])

# 5. Question & Answer Prompt
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a factual Q&A assistant."),
    (
        "human",
        """Context information:
\"\"\"
{context}
\"\"\"

Question: {question}

Provide a direct answer strictly based on the context above.""",
    ),
])
