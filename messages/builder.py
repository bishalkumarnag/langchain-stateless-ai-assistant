"""
builder.py - LangChain Message Construction & Formatting

Contains dedicated logic to construct, validate, and assemble official
LangChain message objects (HumanMessage, AIMessage, SystemMessage).
"""

from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


def create_human_message(content: str) -> HumanMessage:
    """
    Constructs a validated LangChain HumanMessage from user input.
    """
    if not content or not content.strip():
        raise ValueError("Human message content cannot be empty.")
    return HumanMessage(content=content.strip())


def create_system_message(content: str) -> SystemMessage:
    """
    Constructs a validated LangChain SystemMessage from a persona or directive.
    """
    if not content or not content.strip():
        raise ValueError("System message content cannot be empty.")
    return SystemMessage(content=content.strip())


def create_ai_message(content: str) -> AIMessage:
    """
    Constructs a LangChain AIMessage representing assistant completion.
    """
    return AIMessage(content=content)


def format_stateless_turn(
    user_input: str,
    system_instruction: Optional[str] = None,
) -> List[BaseMessage]:
    """
    Assembles an isolated list of LangChain messages for a single, stateless turn.
    
    Notice: Only the current system instruction and current user input are included.
    No prior chat history is preserved or appended.

    Args:
        user_input (str): The current query from the user.
        system_instruction (Optional[str]): Optional system prompt.

    Returns:
        List[BaseMessage]: [SystemMessage (optional), HumanMessage]
    """
    messages: List[BaseMessage] = []

    if system_instruction:
        messages.append(create_system_message(system_instruction))

    messages.append(create_human_message(user_input))

    return messages
