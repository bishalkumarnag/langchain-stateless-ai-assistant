# Stateless Conversational AI Assistant (LangChain)

A modular, production-ready Python conversational AI assistant built **100% using standard modern LangChain** (`langchain`, `langchain-core`, `langchain-google-genai`, `langchain-openai`).

It is designed intentionally **without preserved context support** (stateless, single-turn per query) using **LangChain Expression Language (LCEL)**, and architected so that **switching to any other LLM requires editing only ONE single file**.

---

## 🎯 The 5 Core Pillars (All Standard LangChain)

| Component | Status | Official LangChain Class | Module in Project |
| :--- | :---: | :--- | :--- |
| **LLM** | `[✓]` | `BaseChatModel` via `init_chat_model` | [`llm/model_factory.py`](./llm/model_factory.py) |
| **Parameters** | `[✓]` | Model kwargs & native `.bind(temperature=...)` | [`llm/model_factory.py`](./llm/model_factory.py) |
| **Prompting** | `[✓]` | `ChatPromptTemplate` & prompt builder | [`prompts/builder.py`](./prompts/builder.py) |
| **Output Parser** | `[✓]` | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` | [`parsers/factory.py`](./parsers/factory.py) |
| **Message Types** | `[✓]` | `HumanMessage`, `AIMessage`, `SystemMessage` & turn builder | [`messages/builder.py`](./messages/builder.py) |

---

## 🔄 How to Switch to Any Other LLM (In 1 File)

All LLM instantiation is encapsulated in **[`llm/model_factory.py`](./llm/model_factory.py)** using LangChain's official **`init_chat_model()`**. The rest of the codebase interacts strictly with LangChain's standard `BaseChatModel` interface.

### Option A: Change 1 line in `.env` (No code changes!)
```env
# Switch from Google Gemini to OpenAI:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your_key
OPENAI_DEFAULT_MODEL=gpt-4o-mini
```

### Option B: Modify `llm/model_factory.py` directly:
In [`llm/model_factory.py`](./llm/model_factory.py), `init_chat_model()` supports any provider:
```python
# To switch to Anthropic Claude:
return init_chat_model("claude-3-5-sonnet-20241022", model_provider="anthropic", api_key=...)

# To switch to Local Ollama:
return init_chat_model("llama3", model_provider="ollama")
```
Every chain, prompt, parser, and UI component continues working seamlessly with **zero other file edits**.

---

## ⚡ The LangChain LCEL Pipeline

This assistant uses **LangChain Expression Language (LCEL)**:
```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": user_input})
```

### Why "Without Preserved Context" (Stateless)?
In multi-turn chat applications, memory buffers (`ChatMessageHistory`) cause transcripts to grow indefinitely, resulting in exponential token costs and context drift.

In this assistant:
- **No memory buffer is attached to the chain.**
- Each `.invoke()` executes an isolated, clean single-turn request.
- Messages leave scope immediately after the response is returned, guaranteeing zero context leakage.

---

## 📁 Project Structure & Modular Architecture

```
langchain-stateless-ai-assistant/
├── config/
│   └── settings.py               # Credentials & active provider selection
├── llm/
│   ├── model_factory.py          # ⭐ Uses LangChain's official init_chat_model()
│   └── __init__.py
├── messages/
│   ├── builder.py                # Message creation & format_stateless_turn logic
│   └── __init__.py               # Re-exports HumanMessage, AIMessage, SystemMessage
├── prompts/
│   ├── builder.py                # create_chat_prompt & prompt assembly logic
│   ├── library.py                # Predefined LangChain ChatPromptTemplates
│   └── __init__.py
├── parsers/
│   ├── factory.py                # get_string_parser, get_json_parser, get_pydantic_parser
│   └── __init__.py               # Re-exports LangChain Output Parsers
├── assistant/
│   ├── conversational_ai.py      # StatelessLangChainAssistant (orchestrates modules above)
│   └── __init__.py
├── tests/
│   └── test_all.py               # Unit test suite (7 tests passing)
├── architecture_diagram.png      # High-level architecture & sequence diagram
├── demo_all_components.py        # 5-step walkthrough of LangChain components
├── main.py                       # Interactive CLI using LangChain
├── requirements.txt              # Official LangChain dependencies
├── .env.example                  # Template configuration for API keys
├── .gitignore                    # Ensures .env & virtual environment are never committed
├── README.md                     # Project documentation & setup instructions
└── HOW_IT_WORKS.md               # Technical deep dive with architecture diagram
```

---

## 🚀 Quickstart & Setup Guide

If you clone this project from GitHub, follow these simple steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell):
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install All Dependencies
Install the required LangChain libraries and packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key (Security Guaranteed)
> [!NOTE]
> **Your `.env` file is protected by `.gitignore`** and will **never** be pushed to GitHub. Each user supplies their own API keys via local configuration.

1. Copy the provided `.env.example` template to create your local `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and paste your Google Gemini or OpenAI API key:
   ```env
   LLM_PROVIDER=google
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_DEFAULT_MODEL=gemini-3.5-flash-lite
   ```
---

## 💻 Running the Application

### 1. Run the Interactive CLI Assistant
```bash
python main.py
```
**Interactive CLI Commands:**
- Type any question to interact with the model via LangChain.
- `/provider` — View current active provider, class name, and model ID.
- `/system <prompt>` — Update system prompt instruction dynamically.
- `/tokens` — Toggle token metadata display.
- `/exit` — Quit.

### 2. Run the All-Component Walkthrough
```bash
python demo_all_components.py
```
Demonstrates Messages, Model Factory, ChatPromptTemplates, Output Parsers, and live LCEL chain execution across all 5 required areas.

### 3. Run the Unit Test Suite
```bash
python -m unittest tests/test_all.py
```
Output:
```
.......
----------------------------------------------------------------------
Ran 7 tests in 0.035s

OK
```

---

## 👤 Author

**Bishal Kumar Nag**
- GitHub: [@bishalkumarnag](https://github.com/bishalkumarnag)
- LinkedIn: [Bishal Kumar Nag](https://www.linkedin.com/in/bishal-kumar-nag/)


