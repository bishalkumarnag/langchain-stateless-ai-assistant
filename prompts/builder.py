"""
builder.py - LangChain Prompt Construction Logic

Provides helper functions and builders to construct official LangChain
ChatPromptTemplate and PromptTemplate instances.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def create_chat_prompt(
    system_instruction: str,
    user_input_placeholder: str = "{input}",
) -> ChatPromptTemplate:
    """
    Constructs a standard two-role LangChain ChatPromptTemplate:
      1. SystemMessagePromptTemplate: Developer persona / instruction
      2. HumanMessagePromptTemplate: User question / query placeholder

    Args:
        system_instruction (str): Persona or directives for the model.
        user_input_placeholder (str): Input variable template (default: "{input}").

    Returns:
        ChatPromptTemplate: An executable LangChain ChatPromptTemplate.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", user_input_placeholder),
    ])


def create_prompt_with_instructions(
    system_instruction: str,
    format_instructions: str,
    user_input_placeholder: str = "{input}",
) -> ChatPromptTemplate:
    """
    Builds a LangChain ChatPromptTemplate with output schema format instructions
    injected into the system directive (commonly used for JSON and Pydantic parsers).
    """
    augmented_system = f"{system_instruction}\n\nFormat Instructions:\n{format_instructions}"
    return create_chat_prompt(
        system_instruction=augmented_system,
        user_input_placeholder=user_input_placeholder,
    )
