
from langchain_core.prompts import ChatPromptTemplate


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
