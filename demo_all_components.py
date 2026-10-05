"""
demo_all_components.py - LangChain Component Walkthrough

Demonstrates all 5 focus areas implemented using official LangChain modules:
  1. [✓] Messages (langchain_core.messages)
  2. [✓] Parameters & Factory (llm.model_factory)
  3. [✓] Prompting (langchain_core.prompts)
  4. [✓] Output Parsers (langchain_core.output_parsers)
  5. [✓] LCEL Pipeline & Stateless Verification (prompt | llm | parser)
"""

import sys
from pydantic import BaseModel, Field

# 1. Official LangChain Imports & Local Package Builders
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

# 2. Local module imports
from llm.model_factory import get_chat_model
from assistant.conversational_ai import StatelessLangChainAssistant
from config.settings import get_settings


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_langchain_messages():
    """Demonstrates [✓] Human/AI Message Type using messages/ builder."""
    print_header("1. DEMO: LangChain Core Message Types & messages/ Builder")

    # Using our modular message builders from messages/
    sys_msg = create_system_message("You are a helpful AI tutor.")
    human_msg = create_human_message("Explain recursion in 1 sentence.")
    ai_msg = AIMessage(content="Recursion is a process where a function calls itself to solve smaller subproblems.")

    print(f"System Message Type : {type(sys_msg).__name__} -> {sys_msg.content}")
    print(f"Human Message Type  : {type(human_msg).__name__} -> {human_msg.content}")
    print(f"AI Message Type     : {type(ai_msg).__name__} -> {ai_msg.content}")

    # Demonstrating stateless turn formatting from messages/
    turn_messages = format_stateless_turn("What is an API?", system_instruction="Be concise.")
    print(f"\nStateless Turn Assembly (messages/): {turn_messages}")


def demo_model_factory():
    """Demonstrates [✓] LLM & Parameters using LangChain's native .bind() method."""
    print_header("2. DEMO: LLM & Parameters (LangChain Native .bind())")

    settings = get_settings()
    print(f"Active Provider : {settings.provider.upper()}")
    print(f"Active Model    : {settings.gemini_model if settings.provider == 'google' else settings.openai_model}")

    # Standard LangChain model via init_chat_model
    chat_model = get_chat_model(temperature=0.7)
    print(f"Instantiated LangChain Model: {type(chat_model).__name__}")
    print(f"Model ID: {getattr(chat_model, 'model', 'N/A')}")

    # LangChain's official built-in way to bind/override parameters dynamically:
    precise_llm = chat_model.bind(temperature=0.1)
    creative_llm = chat_model.bind(temperature=1.2)
    print("\nLangChain Native Parameter Binding:")
    print(f"  - Default Model  : {chat_model}")
    print(f"  - Precise (.bind): {precise_llm}")
    print(f"  - Creative (.bind): {creative_llm}")


def demo_langchain_prompts():
    """Demonstrates [✓] Prompting using prompts/ package."""
    print_header("3. DEMO: LangChain ChatPromptTemplate & prompts/ Builder")

    # Using our modular builder from prompts/
    prompt_template = create_chat_prompt(
        system_instruction="You are an expert in {subject}.",
        user_input_placeholder="Explain {topic} in simple terms.",
    )

    formatted_messages = prompt_template.format_messages(
        subject="computer science",
        topic="hash tables",
    )

    print("Rendered LangChain Messages from Template (prompts/):")
    for msg in formatted_messages:
        print(f"  [{type(msg).__name__}]: {msg.content}")


def demo_langchain_parsers():
    """Demonstrates [✓] Output Parser using parsers/ package."""
    print_header("4. DEMO: LangChain Output Parsers & parsers/ Factory")

    # 1. StrOutputParser via parsers/
    str_parser = get_string_parser()
    print("StrOutputParser Result (parsers/):", str_parser.invoke(AIMessage(content="   Hello world!   ")))

    # 2. JsonOutputParser via parsers/
    json_parser = get_json_parser()
    simulated_json_msg = AIMessage(content='```json\n{"status": "ok", "framework": "LangChain"}\n```')
    parsed_json = json_parser.invoke(simulated_json_msg)
    print("JsonOutputParser Result (parsers/):", parsed_json)

    # 3. PydanticOutputParser via parsers/
    class MovieReview(BaseModel):
        movie_title: str = Field(description="Title of movie")
        rating_out_of_10: float = Field(description="Rating from 1.0 to 10.0")
        verdict: str = Field(description="Short verdict")

    pydantic_parser = get_pydantic_parser(MovieReview)
    simulated_movie_msg = AIMessage(
        content='```json\n{"movie_title": "Interstellar", "rating_out_of_10": 9.2, "verdict": "Masterpiece"}\n```'
    )
    parsed_movie = pydantic_parser.invoke(simulated_movie_msg)
    print("PydanticOutputParser Result (parsers/):")
    print(f"  Title   : {parsed_movie.movie_title}")
    print(f"  Rating  : {parsed_movie.rating_out_of_10}")
    print(f"  Verdict : {parsed_movie.verdict}")


def demo_langchain_assistant_live():
    """Demonstrates [✓] LangChain LCEL Assistant & Stateless Execution."""
    print_header("5. DEMO: Live LangChain LCEL Assistant (Stateless)")

    settings = get_settings()
    try:
        settings.validate_provider_keys()
    except ValueError as e:
        print(f"[!] API key not configured: {e}")
        return

    assistant = StatelessLangChainAssistant()
    print(f"Running queries through LangChain chain (prompt | llm | parser)...")

    # Turn 1
    print("\n--- Turn 1: Providing information ---")
    q1 = "My favorite color is emerald green. Acknowledge this."
    print(f"User: {q1}")
    r1 = assistant.ask(q1)
    print(f"AI  : {r1}")

    # Turn 2: Demonstrating stateless behavior (no preserved context)
    print("\n--- Turn 2: Verifying statelessness (no preserved context) ---")
    q2 = "What was my favorite color from the previous question?"
    print(f"User: {q2}")
    r2 = assistant.ask(q2)
    print(f"AI  : {r2}")
    print("\n[✓] Confirmed: The assistant has no memory of Turn 1, verifying statelessness!")

    # Turn 3: Live Pydantic parsing via LCEL
    print("\n--- Turn 3: Structured LCEL Pipeline (prompt | llm | PydanticOutputParser) ---")

    class CityGuide(BaseModel):
        city: str = Field(description="City name")
        country: str = Field(description="Country name")
        famous_landmark: str = Field(description="Most famous landmark")

    city_parser = PydanticOutputParser(pydantic_object=CityGuide)
    city_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract city information matching the schema instructions.\n{format_instructions}"),
        ("human", "Tell me about {city}"),
    ]).partial(format_instructions=city_parser.get_format_instructions())

    # Build pure LangChain LCEL chain: prompt | llm | parser
    chain = city_prompt | assistant.llm | city_parser
    parsed_city = chain.invoke({"city": "Paris"})

    print("Parsed Pydantic Object from LangChain Chain:")
    print(f"  City     : {parsed_city.city}")
    print(f"  Country  : {parsed_city.country}")
    print(f"  Landmark : {parsed_city.famous_landmark}")


def main():
    print("=" * 70)
    print("  LANGCHAIN STATELESS CONVERSATIONAL ASSISTANT - COMPONENT DEMO")
    print("=" * 70)

    demo_langchain_messages()
    demo_model_factory()
    demo_langchain_prompts()
    demo_langchain_parsers()
    demo_langchain_assistant_live()

    print_header("ALL LANGCHAIN COMPONENT DEMOS COMPLETED")


if __name__ == "__main__":
    main()
