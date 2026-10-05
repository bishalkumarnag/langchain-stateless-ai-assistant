"""
prompts package - LangChain Prompt Construction & Templates

Exports:
- Prompt builder functions: create_chat_prompt, create_prompt_with_instructions
- Official LangChain classes: ChatPromptTemplate, PromptTemplate
- Predefined templates: CONVERSATIONAL_PROMPT, CODE_EXPLAINER_PROMPT, etc.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from .builder import create_chat_prompt, create_prompt_with_instructions
from .library import (
    DEFAULT_SYSTEM_PROMPT,
    CONVERSATIONAL_PROMPT,
    CODE_EXPLAINER_PROMPT,
    TEXT_SUMMARIZER_PROMPT,
    QA_PROMPT,
)

__all__ = [
    "create_chat_prompt",
    "create_prompt_with_instructions",
    "ChatPromptTemplate",
    "PromptTemplate",
    "SystemMessagePromptTemplate",
    "HumanMessagePromptTemplate",
    "DEFAULT_SYSTEM_PROMPT",
    "CONVERSATIONAL_PROMPT",
    "CODE_EXPLAINER_PROMPT",
    "TEXT_SUMMARIZER_PROMPT",
    "QA_PROMPT",
]
