"""
main.py - Interactive CLI for LangChain Conversational Assistant (Stateless)

Built exclusively with official LangChain primitives:
  - ChatPromptTemplate (langchain_core.prompts)
  - BaseChatModel via init_chat_model (langchain.chat_models)
  - StrOutputParser (langchain_core.output_parsers)
  - LCEL Pipeline: prompt | llm | parser
"""

import sys
from config.settings import get_settings
from llm.model_factory import get_chat_model
from assistant.conversational_ai import StatelessLangChainAssistant


def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║         LANGCHAIN STATELESS CONVERSATIONAL ASSISTANT               ║
║                                                                    ║
║  Core Pillars (All Official LangChain Built-in Standards):         ║
║  [✓] LLM: init_chat_model (langchain.chat_models)                  ║
║  [✓] Parameters: Built-in model parameters / .bind()               ║
║  [✓] Prompting: ChatPromptTemplate (langchain_core.prompts)        ║
║  [✓] Output Parser: StrOutputParser (langchain_core.output_parsers)║
║  [✓] Messages: HumanMessage, AIMessage, SystemMessage              ║
║                                                                    ║
║  Execution: Standard LangChain LCEL (prompt | llm | parser)        ║
║  State: Stateless (no conversation memory buffer attached)         ║
╚════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("Commands:")
    print("  /help            - Show this command reference")
    print("  /provider        - Show active LLM provider and model")
    print("  /system <text>   - Update system prompt instruction")
    print("  /exit or /quit   - Exit the application\n")


def main():
    print_banner()

    settings = get_settings()
    try:
        settings.validate_provider_keys()
    except ValueError as e:
        print(f"⚠️  {e}\n")
        sys.exit(1)

    # 1. Initialize LangChain model via official init_chat_model
    llm = get_chat_model()

    # 2. Initialize Stateless LangChain Assistant
    assistant = StatelessLangChainAssistant(llm=llm)

    active_model = settings.gemini_model if settings.provider == "google" else settings.openai_model
    print(f"✅ LangChain Chat Model Initialized: {type(llm).__name__} ({active_model})")
    print(f"🌐 Active Provider: {settings.provider.upper()} (Change LLM_PROVIDER in .env to switch)")
    print("Type your message below and press Enter:\n" + "-" * 70)

    system_prompt = None

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Goodbye!")
                break

            elif user_input.lower() == "/help":
                print_banner()
                continue

            elif user_input.lower() == "/provider":
                print(f"Active Provider : {settings.provider.upper()}")
                print(f"Chat Model Class: {type(llm).__name__}")
                print(f"Active Model ID : {active_model}")
                print("💡 To switch to OpenAI, set LLM_PROVIDER=openai in .env")
                continue

            elif user_input.lower().startswith("/system "):
                system_prompt = user_input[8:].strip()
                print(f"System directive updated to:\n\"{system_prompt}\"")
                continue

            # Standard LangChain LCEL Invocation:
            # chain = prompt | llm | StrOutputParser()
            # output is parsed directly through StrOutputParser
            response = assistant.ask(
                user_input=user_input,
                system_prompt=system_prompt,
            )

            print(f"\n{settings.provider.title()} > {response}")

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during LangChain generation: {e}")


if __name__ == "__main__":
    main()
