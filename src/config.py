
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict, List
import operator
import streamlit as st

# ============================================================================
# State Definitions used in the LangGraph graph
# ============================================================================

class BusinessDetail(TypedDict):
    """
    Represents a business detail with name, URL, summary, and city.

    Attributes:
        name: The name of the business.
        url: The website URL of the business.
        summary: website content (summary of the business).
        city: The target city where the business is located or operates.
    """
    name: str
    url: str
    summary: str
    city: str
    
class GraphState(TypedDict):
    """
    State representation for the agent LangGraph workflow.

    Attributes:
        query: The search query.
        business_name: The name of the business
        business_type: The industry or type of business.
        cities: The list of cities used in the search.
        current_city_index: The current city index.
        no_of_cities: The total number of cities.
        current_city: The current city being processed.
        target_cities: The list of target cities for the business.
        province: The target province/state.
        country: The target country.
        max_sites: Maximum number of sites to return
        search_results: List of search results
        landing_page_draft: The landing page draft
        draft_option_no: The number of draft options to include
        last_critique: The last critique of the landing page draft
        approval_status: Current approval status
        revision_count: Current revision count
        max_revisions: Maximum number of revisions
    """
    query: str
    business_name: str
    business_type: str
    cities: List[str]
    current_city_index: int
    no_of_cities: int
    current_city: str
    target_cities: str
    province: str
    country: str
    max_sites: int
    search_results: Annotated[List[BusinessDetail], operator.add]
    landing_page_draft: str
    draft_option_no: int
    last_critique: str
    approval_status: str
    revision_count: int
    max_revisions: int


def get_api_key(key_name):
    """
    Retrieves an API key from Streamlit secrets or environment variables.

    This function first attempts to fetch the requested key from Streamlit's
    secrets management system (`st.secrets`). If the key is not found there,
    it falls back to reading the value from environment variables via
    `os.getenv`.

    This allows the function to work seamlessly in both:
    - Streamlit Cloud / local Streamlit apps (using `st.secrets`)
    - Traditional Python environments (using `.env` or system environment variables)

    Args:
        key_name (str): The name of the secret key to retrieve (e.g., "TAVILY_API_KEY").

    Returns:
        str | None: The value of the API key if found, otherwise None.
    """
    
    return st.secrets.get(
        key_name,
        os.getenv(key_name)
)

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
        "Ollama": ("OLLAMA_API_KEY", "langchain_ollama", "ChatOllama"),
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
    # Ollama is local and does not require an API key
    if llm_provider == "Ollama":
        api_key = "local-no-key-required"
    else:
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

tavily_api_key = st.secrets["TAVILY_API_KEY"]




