from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
import os


def create_langchain_llm(
    llm_provider: str = "OpenAI",
    model_name: str = None,
    temperature: float = 0.0
) -> BaseChatModel:
    """
    Factory function that creates and configures a LangChain chat model
    based on the selected LLM provider.

    Supported providers:
    - OpenAI
    - Gemini
    - Groq
    - HuggingFace
    - Ollama

    This function handles:
    1. Provider validation
    2. Environment variable setup for LangChain compatibility
    3. Provider-specific model initialization
    4. Special handling for HuggingFace and Ollama backends

    Returns:
        A configured LangChain BaseChatModel instance.
    """

    # -------------------------------------------------------
    # 1. Provider registry
    # -------------------------------------------------------
    providers = {
        "OpenAI": ("OPENAI_API_KEY", "langchain_openai", "ChatOpenAI"),
        "Gemini": ("GEMINI_API_KEY", "langchain_google_genai", "ChatGoogleGenerativeAI"),
        "Groq": ("GROQ_API_KEY", "langchain_groq", "ChatGroq"),
        "HuggingFace": ("HUGGINGFACE_API_KEY", "langchain_huggingface", "ChatHuggingFace"),
    }

    # -------------------------------------------------------
    # 2. Validate provider
    # -------------------------------------------------------
    if llm_provider not in providers:
        raise ValueError(f"Unsupported provider: {llm_provider}")

    api_key_name, module_name, class_name = providers[llm_provider]

    # -------------------------------------------------------
    # 3. Load API key
    # -------------------------------------------------------
    api_key = get_api_key(api_key_name)

    if not api_key:
        raise ValueError(
            f"Missing environment file or API key: {api_key_name}"
        )

    # -------------------------------------------------------
    # 4. Set environment variables for LangChain internals
    # -------------------------------------------------------
    if llm_provider == "Gemini":
        os.environ["GOOGLE_API_KEY"] = api_key

    elif llm_provider == "HuggingFace":
        os.environ["HF_TOKEN"] = api_key

    else:
        os.environ[api_key_name] = api_key

    # -------------------------------------------------------
    # 5. HuggingFace special handling
    # -------------------------------------------------------
    if llm_provider == "HuggingFace":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        is_greedy = (temperature == 0.0)

        llm_engine = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            huggingfacehub_api_token=api_key,
            max_new_tokens=512,
            temperature=None if is_greedy else temperature,
            do_sample=not is_greedy,
            model_kwargs={}
        )

        return ChatHuggingFace(
            llm=llm_engine,
            streaming=False
        )

    # -------------------------------------------------------
    # 6. Ollama local model handling
    # -------------------------------------------------------
    if llm_provider == "Ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    # -------------------------------------------------------
    # 7. Dynamic import for other providers
    # -------------------------------------------------------
    try:
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)

    except ImportError:
        raise ImportError(f"Please install {module_name}")

    # -------------------------------------------------------
    # 8. Standard model initialization
    # -------------------------------------------------------
    return model_class(
        model=model_name,
        temperature=temperature
    )


def create_agent(llm, tools: list, system_prompt: str):
    """
    Creates a LangChain agent composed of:
    - A system prompt
    - A message placeholder for runtime input
    - An optional tool-binding layer

    This function builds a simple prompt → model (or tool-enabled model) pipeline.

    Behavior:
    - If tools are provided:
        The LLM is bound to tools using `bind_tools()`, enabling tool calling.
    - If no tools are provided:
        The LLM is used directly without tool augmentation.

    Returns:
        (agent_chain, tools)

    Notes:
        - The agent is a LangChain Runnable (LCEL chain)
        - It does NOT execute tools itself
        - It does NOT manage memory or state
    """

    # -------------------------------------------------------
    # 1. Build prompt template
    # -------------------------------------------------------
    # Create a chat prompt template using LangChain's ChatPromptTemplate
    # This defines the structure of messages that will be sent to the LLM.
    # System message: sets the behavior, role, and instructions for the model
    # Human message: dynamic input placeholder that will be filled at runtime
    # when invoking the chain using {"input": user_prompt}
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # -------------------------------------------------------
    # 2. Attach tools if provided
    # -------------------------------------------------------
    # Tool binding enables function-calling capability
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        agent = prompt | llm_with_tools

    # -------------------------------------------------------
    # 3. Standard LLM pipeline (no tools)
    # -------------------------------------------------------
    else:
        agent = prompt | llm

    return agent, tools
