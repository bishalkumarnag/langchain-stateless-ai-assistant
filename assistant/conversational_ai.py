"""
conversational_ai.py - LangChain LCEL Conversational Assistant (Stateless)

Orchestrates the modular components:
  - LLM: from llm.model_factory import get_chat_model
  - Prompts: from prompts.builder import create_chat_prompt
             from prompts.library import CONVERSATIONAL_PROMPT
  - Parsers: from parsers import get_string_parser, BaseOutputParser
  - Messages: from messages import format_stateless_turn, BaseMessage

Pipeline:
    chain = prompt | llm | parser
    return chain.invoke({"input": user_input})

Stateless by Design:
    No conversation memory buffer is attached. Each query executes in isolation.
"""

from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable

# 1. Imports from our modular project packages
from llm.model_factory import get_chat_model
from prompts.builder import create_chat_prompt
from prompts.library import CONVERSATIONAL_PROMPT
from parsers import get_string_parser
from messages import format_stateless_turn, BaseMessage


class StatelessLangChainAssistant:
    """
    Stateless Conversational Assistant orchestrating LangChain components via LCEL.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        parser: Optional[BaseOutputParser] = None,
    ):
        # 1. Initialize LangChain Chat Model from llm/
        self.llm: BaseChatModel = llm or get_chat_model()

        # 2. Initialize LangChain Prompt Template from prompts/
        self.prompt: ChatPromptTemplate = prompt or CONVERSATIONAL_PROMPT

        # 3. Initialize LangChain Output Parser from parsers/
        self.parser: BaseOutputParser = parser or get_string_parser()

        # 4. Standard LangChain LCEL Pipeline: prompt | llm | parser
        self.chain: Runnable = self.prompt | self.llm | self.parser

    def ask(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        parser: Optional[BaseOutputParser] = None,
    ) -> Any:
        """
        Executes a single conversational turn through the LangChain LCEL chain:
            chain = prompt | llm | parser
            return chain.invoke({"input": user_input})
        """
        # Consume prompt builder from prompts/ if custom system prompt is provided
        active_prompt = (
            create_chat_prompt(system_instruction=system_prompt)
            if system_prompt
            else self.prompt
        )

        # Consume parser from parsers/ (or default)
        active_parser = parser or self.parser

        # Compose and execute the pure LangChain LCEL chain
        chain = active_prompt | self.llm | active_parser
        return chain.invoke({"input": user_input})

    def run_messages(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Alternative execution method consuming the message builder from messages/:
        Constructs [SystemMessage, HumanMessage] and passes directly to the LLM.
        """
        # Consume message construction logic from messages/
        message_list: List[BaseMessage] = format_stateless_turn(
            user_input=user_input,
            system_instruction=system_prompt,
        )

        # Execute model directly with LangChain messages and parse with active parser
        response = self.llm.invoke(message_list)
        return self.parser.invoke(response)

    def run_template(
        self,
        template: ChatPromptTemplate,
        variables: Dict[str, Any],
        parser: Optional[BaseOutputParser] = None,
    ) -> Any:
        """
        Executes any arbitrary LangChain ChatPromptTemplate with LCEL.
        """
        active_parser = parser or self.parser
        chain = template | self.llm | active_parser
        return chain.invoke(variables)
