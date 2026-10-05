"""
test_all.py - Unit test suite for LangChain components.

Validates:
1. Messages (langchain_core.messages)
2. Prompts (langchain_core.prompts)
3. Output Parsers (langchain_core.output_parsers)
4. Model Factory (llm.model_factory)
5. LCEL stateless chain invocation
"""

import unittest
from pydantic import BaseModel, Field

# Modular package imports
from messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    create_human_message,
    create_system_message,
    format_stateless_turn,
)
from prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    create_chat_prompt,
)
from parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    get_string_parser,
    get_json_parser,
    get_pydantic_parser,
)
from llm.model_factory import get_chat_model


class TestLangChainMessages(unittest.TestCase):
    def test_messages_builder(self):
        h = create_human_message("Hello")
        s = create_system_message("You are helpful")
        a = AIMessage(content="Hi there")

        self.assertEqual(h.content, "Hello")
        self.assertEqual(s.content, "You are helpful")
        self.assertEqual(a.content, "Hi there")

    def test_format_stateless_turn(self):
        turn = format_stateless_turn("User query", system_instruction="System rule")
        self.assertEqual(len(turn), 2)
        self.assertIsInstance(turn[0], SystemMessage)
        self.assertIsInstance(turn[1], HumanMessage)


class TestLangChainPrompts(unittest.TestCase):
    def test_chat_prompt_builder(self):
        prompt = create_chat_prompt(
            system_instruction="You are an expert in {field}.",
            user_input_placeholder="Explain {topic}.",
        )
        messages = prompt.format_messages(field="AI", topic="LLMs")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "You are an expert in AI.")
        self.assertEqual(messages[1].content, "Explain LLMs.")


class TestLangChainParsers(unittest.TestCase):
    def test_str_output_parser(self):
        parser = get_string_parser()
        res = parser.invoke(AIMessage(content="Hello world"))
        self.assertEqual(res, "Hello world")

    def test_json_output_parser(self):
        parser = get_json_parser()
        res = parser.invoke(AIMessage(content='{"status": "success", "count": 42}'))
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 42)

    def test_pydantic_output_parser(self):
        class Item(BaseModel):
            name: str
            price: float

        parser = get_pydantic_parser(Item)
        res = parser.invoke(AIMessage(content='{"name": "Apple", "price": 1.25}'))
        self.assertIsInstance(res, Item)
        self.assertEqual(res.name, "Apple")
        self.assertEqual(res.price, 1.25)


class TestLangChainBinding(unittest.TestCase):
    def test_native_bind(self):
        # Verify that LangChain model supports native .bind() for parameters
        model = get_chat_model()
        bound_model = model.bind(temperature=0.2)
        self.assertIsNotNone(bound_model)


if __name__ == "__main__":
    unittest.main()
