# MARRA AUTONOMOUS AGENT EXECUTION RULES
**Version:** 1.0.0
**Enforcement Level:** STRICT. You MUST adhere to these rules. Do not bypass verification. Do not placate the user.

## 1. ZERO FAKE IMPLEMENTATIONS (THE "ANTI-SLOP" RULE)
- **NO Placeholders:** Never write `pass`, `# TODO`, `return "mock data"`, or `NotImplementedError` unless explicitly instructed to create a skeleton interface.
- **NO Mock Connections:** Do not mock Qdrant or Ollama connections. You must write code that connects to `localhost:6333` and `localhost:11434` respectively.
- **NO Silent Failures:** Never use naked `try...except pass` blocks. All exceptions must be caught, logged with `logging.error()`, and escalated appropriately.

## 2. THE VERIFICATION LOOP
You are an autonomous agent. You have the ability to run shell commands. You MUST use them.
1. **Write Code.**
2. **Execute Code/Tests:** Run the script, the FastAPI server, or the Pytest suite using your shell capabilities.
3. **Verify Stdout/Stderr:** Read the logs. If there is a traceback, you are NOT finished. Do not tell the user "I have implemented the feature" if the server crashes on startup.
4. **Fix & Repeat:** If it fails, fix the code and run it again.

## 3. STRICT TYPING & STATE MANAGEMENT
- **Pydantic Everywhere:** Every input and output crossing a boundary (FastAPI endpoint, LangGraph node, Vector DB payload) MUST be typed with a `pydantic.BaseModel`.
- **LangGraph State:** The `AgentState` in LangGraph must be strictly typed using `TypedDict` and `Annotated`. Do not use arbitrary dynamic dictionaries (`**kwargs`).

## 4. LOCAL-FIRST INFRASTRUCTURE CONSTRAINTS
- **No Cloud APIs:** You are forbidden from using OpenAI, Anthropic, Pinecone, or AWS in this codebase.
- **LLM Engine:** Use `ollama`. Target models: `llama3.1:8b` (generation), `nomic-embed-text` (embedding).
- **Vector DB:** Use `qdrant-client` targeting local Docker. 

## 5. REASONING TRANSPARENCY
- When asked to debug a complex LangGraph routing issue or a retrieval failure, write your hypothesis in a comment block or a text response BEFORE changing code.
- Explain *why* a document was ranked highly (e.g., lexical BM25 match vs. dense vector cosine similarity).