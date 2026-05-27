
from src.config import GraphState
from src.tools import search_website_content
from src.agent import create_agent
import ast
import json


def search_node(state: GraphState):
    """
    Search node for LangGraph pipeline.

    This node:
    1. Extracts search parameters from the current graph state
    2. Calls the Tavily-based website search tool
    3. Attempts to parse and structure raw results into Python objects
    4. Injects metadata (city context) into each result
    5. Returns updated state with structured search results

    The node also advances the city index: current_city_index, to enable iterative multi-city search.
    """

    # -----------------------------
    # 1. Extract state variables
    # -----------------------------
    query = state["query"]
    b_type = state["business_type"]
    cities = state["cities"]
    current_city_index = state["current_city_index"]
    no_of_cities = state["no_of_cities"]
    current_city = state["current_city"]
    prov = state["province"]
    country = state["country"]
    max_sites = state["max_sites"]

    # Current city being processed in the loop
    current_city = cities[current_city_index]

    no_of_cities = len(cities)

    # -----------------------------
    # 2. Execute external search tool
    # -----------------------------
    raw_results = search_website_content.invoke({
        "query": query,
        "business_type": b_type,
        "city": current_city,
        "province": prov,
        "country": country,
        "max_sites": max_sites
    })

    # -----------------------------
    # 3. Parse and structure results
    # -------------------------
    try:
        # Convert the string representation of a list into a real Python list
        # We use ast.literal_eval because it's safer for stringified Python objects
        structured_data = ast.literal_eval(raw_results)
        
        # Enrich each result with contextual metadata (city)
        for item in structured_data:
            item['city'] = current_city
            
    except (ValueError, SyntaxError) as e:
        # Fallback if parsing fails (unexpected tool output format)
        print(f"Parsing error: {e}")
        structured_data = [
            {
                "name": "Error",
                "summary": str(raw_results),
                "city": current_city
            }
        ]
    
    # -----------------------------
    # 4. Return updated graph state
    # -----------------------------
    return {
        "search_results": structured_data,
        "current_city": current_city,
        "no_of_cities": no_of_cities,
        # Move to next city for iterative processing
        "current_city_index": current_city_index + 1,
    }


def city_router(state: GraphState) -> str:
    """
    LangGraph router that determines whether additional cities
    still need to be processed.

    The router checks the current city index against the total
    number of cities in the state.

    Returns:
        "continue":
            More cities remain to be processed.

        "next":
            All cities have been processed and the graph
            should transition to the next workflow stage.
    """

    # Check whether all cities have already been processed
    if state["current_city_index"] >= len(state["cities"]):
        return "next"

    # Continue processing remaining cities
    return "continue"


def designer_node(state: GraphState, llm):
    """
    LangGraph node responsible for generating landing page copy drafts.

    This node:
    1. Extracts business and search context from graph state
    2. Formats competitor research data into prompt-safe JSON
    3. Creates a specialized content-generation agent
    4. Generates multiple landing page draft options
    5. Returns generated drafts back into graph state

    The generated content is intended to be:
    - inspired by competitor research
    - stylistically unique
    - human-sounding
    - suitable for website landing pages
    """

    # -----------------------------
    # 1. Extract state variables
    # -----------------------------
    b_type = state["business_type"]
    business_name = state["business_name"]
    draft_option_no = state["draft_option_no"]
    search_results = state["search_results"]

    # -----------------------------
    # 2. Prepare search results
    # -----------------------------
    # Convert structured search results into JSON string format
    search_results_json = json.dumps(search_results)

    # Escape curly braces so JSON is not interpreted as
    # Python format-template placeholders during .format()
    escaped_results = search_results_json.replace("{", "{{").replace("}", "}}")

    # -----------------------------
    # 3. Build system prompt
    # -----------------------------
    system_prompt_template = """You are a professional Website Content Generator.
    
    You use the information from the web to generate a landing page description for 
    {b_type} businesses.
    """
    
    system_prompt = system_prompt_template.format(b_type=b_type)
    
    # -----------------------------
    # 4. Build user prompt
    # -----------------------------
    user_prompt_template = """
    You are given a list of websites in JSON format within
    {search_results} for a particular {b_type} business type.

    Each item contains:
    - name
    - url
    - summary
    - city

    Requirements:

    1. Focus ONLY on the text within the 'summary' fields.
       Ignore business names and URLs.

    2. Use the combined summaries as inspiration for creating
       landing page content for {business_name}.

    3. Generate original content.
       Do NOT copy any wording directly from source material.

    4. Include product/service descriptions.

    5. The tone should be:
       - classy
       - fun
       - professional
       - friendly
       - natural and human-sounding

    6. Avoid generic AI-sounding marketing language.

    7. Generate {draft_option_no} draft options.

    8. Each option should contain:
       - one paragraph
       - approximately 3 to 4 lines
    """

    user_prompt = user_prompt_template.format(
      search_results=escaped_results,
      b_type=b_type,
      business_name=business_name,
      draft_option_no=draft_option_no
    )
    
    # -----------------------------
    # 5. Create specialized agent
    # -----------------------------
    designer_agent, _ = create_agent(llm, [], system_prompt)

    # -----------------------------
    # 6. Generate landing page drafts
    # -----------------------------
    response = designer_agent.invoke({
        "input": user_prompt
    })

    # -----------------------------
    # 7. Return updated graph state
    # -----------------------------
    return {"landing_page_draft": response.content}


def editor_node(state: GraphState, llm):
    # -----------------------------
    # 1. Extract state variables
    # -----------------------------
    b_type = state["business_type"]
    business_name = state["business_name"]
    draft=state["landing_page_draft"]
    draft_option_no = state["draft_option_no"]

    # -----------------------------
    # 2. Build system prompt
    # -----------------------------
    system_prompt = """You are a professional Editor that provides constructive feedback 
    for website content provided.
    """

    # -----------------------------
    # 3. Create editor agent
    # -----------------------------
    editor_agent, _ = create_agent(llm, [], system_prompt)

    
    # -----------------------------
    # 4. Build evaluation prompt
    # -----------------------------
    user_prompt_template = """
    Evaluate the draft for the {draft_option_no}
    different options of website landing page content
    for a {b_type} business needed for {business_name}.

    PREVIOUS DRAFT:
    {draft}

    Check for:

    1. Ensure there are no business names or URLs
       unrelated to {business_name}.

    2. Ensure there are exactly
       {draft_option_no} different landing page options.

    3. Grammar and spelling quality.

    4. Clarity and readability of the writing.

    5. Proper structure:
       - 1 paragraph
       - approximately 3 to 4 lines

    6. Clear product/service descriptions.

    7. Tone consistency:
       - classy
       - fun
       - professional
       - friendly

    If the draft meets all requirements perfectly,
    return ONLY:

    APPROVED

    Do not include any additional text.

    Otherwise, provide specific revision instructions.
    """

    user_prompt = user_prompt_template.format(
        b_type=b_type,
        business_name=business_name,
        draft=draft,
        draft_option_no=draft_option_no
    )

    # -----------------------------
    # 5. Invoke editor agent
    # -----------------------------
    response = editor_agent.invoke({
        "input": user_prompt
    })

    # -----------------------------
    # 6. Determine approval status
    # -----------------------------
    # If the response contains only a short APPROVED message,
    # treat the draft as accepted.
    if ('APPROVED' in response.content.upper()) and (len(response.content.strip()) < 10):
        approval_status = 'APPROVED'
    else:
        approval_status = 'REJECTED'

    # -----------------------------
    # 7. Return updated graph state
    # -----------------------------
    return {
        # Editorial feedback or approval message
        "last_critique": response.content,

        # Workflow approval status used by router logic
        "approval_status": approval_status
    }


def revision_router(state: GraphState) -> str:
    """
    LangGraph router that determines whether the workflow
    should proceed to the next stage or continue revising
    the generated landing page draft.

    Routing logic is based on:
    1. Whether the draft has been approved
    2. Whether the maximum revision limit has been reached

    Returns:
        "next":
            The draft is approved OR the maximum number
            of revisions has been reached.

        "revise":
            Additional revision iterations are required.
    """

    # -----------------------------
    # 1. Check termination conditions
    # -----------------------------
    # Move to the next workflow stage if:
    # - the draft has been approved
    # - OR the revision limit has been reached
    if (state["approval_status"] == "APPROVED") or (state["revision_count"] >= state["max_revisions"]):
        return "next"
    
    # -----------------------------
    # 2. Continue revision loop
    # -----------------------------
    return "revise"


def revision_node(state: GraphState, llm):
    """
    LangGraph node responsible for revising landing page drafts
    based on editorial feedback.

    This node:
    1. Extracts the previous draft and editor feedback from state
    2. Creates a revision-focused agent
    3. Generates an improved version of the landing page content
    4. Increments revision counter
    5. Returns updated draft back into graph state

    The revision process ensures:
    - editorial feedback is addressed
    - content quality improves iteratively
    - tone and structure remain consistent
    """

    # -----------------------------
    # 1. Extract state variables
    # -----------------------------
    b_type = state["business_type"]
    business_name = state["business_name"]
    draft_option_no = state["draft_option_no"]
    previous_draft = state["landing_page_draft"]
    last_critique = state["last_critique"]
    revision_count = state["revision_count"]
    max_revisions = state["max_revisions"]

    # -----------------------------
    # 2. Build system prompt
    # -----------------------------
    system_prompt = """You are an expert content revision specialist.
    
    You improve website content based on editorial feedback,
    ensuring clarity, tone consistency, and quality improvement.
    """
    
    # -----------------------------
    # 3. Create revision agent
    # -----------------------------
    revision_agent, _ = create_agent(llm, [], system_prompt)

    # -----------------------------
    # 4. Build revision prompt
    # -----------------------------
    user_prompt_template = """
    The previous draft for {draft_option_no} options of website
    landing page content for a {b_type} business ({business_name})
    needs improvement.

    PREVIOUS DRAFT:
    {previous_draft}

    EDITOR FEEDBACK:
    {last_critique}

    TASK:
    Revise the landing page content to address all feedback.

    Requirements:
    - Maintain {draft_option_no} distinct options
    - Preserve original intent and structure where possible
    - Ensure clarity, professionalism, and engagement
    - Keep tone: classy, fun, professional, friendly
    - Avoid introducing unrelated content

    Output only the improved version of the content.
    """

    user_prompt = user_prompt_template.format(
      draft_option_no=draft_option_no,
      b_type=b_type,
      business_name=business_name,
      previous_draft=previous_draft,
      last_critique=last_critique
    )

    # -----------------------------
    # 5. Invoke revision agent
    # -----------------------------
    response = revision_agent.invoke({
        "input": user_prompt
    })

    # -----------------------------
    # 6. Return updated state
    # -----------------------------
    return {
        # Updated improved draft
        "landing_page_draft": response.content,
        "max_revisions": max_revisions,

        # Track revision iterations
        "revision_count": revision_count + 1
    }


def seo_node(state: GraphState, llm):
    """
    LangGraph node responsible for transforming final landing page drafts
    into SEO-optimized structured content.

    This node:
    1. Extracts finalized landing page draft and business context from state
    2. Creates an SEO agent
    3. Adds SEO structure (meta tags + markdown hierarchy)
    4. Ensures content is not rewritten, only structurally enhanced
    5. Returns SEO-ready landing page content

    The goal is NOT to rewrite content, but to:
    - improve search engine readability
    - add metadata (title + description)
    - structure content using proper markdown hierarchy
    """

    # -----------------------------
    # 1. Extract state variables
    # -----------------------------
    draft_option_no = state["draft_option_no"]
    b_type = state["business_type"]
    business_name = state["business_name"]
    target_cities = state["target_cities"]
    province = state["province"]
    country = state["country"]
    draft = state["landing_page_draft"]

    # -----------------------------
    # 2. Build system prompt
    # -----------------------------
    system_prompt = """
    You are an expert technical SEO Optimizer.

    Your job is to structure finalized website content so it is:
    - SEO-friendly
    - well-structured in markdown
    - optimized for search engine visibility

    You DO NOT rewrite content meaningfully.
    You only improve structure and add metadata.
    """

    # -----------------------------
    # 3. Create SEO agent
    # -----------------------------
    seo_agent, _ = create_agent(
        llm,
        [],
        system_prompt
    )

    # -----------------------------
    # 4. Build SEO optimization prompt
    # -----------------------------
    user_prompt_template = """
    You are given {draft_option_no} final draft options of website
    landing page content for a {b_type} business named {business_name}.

    BUSINESS LOCATION CONTEXT:
    - Target Cities: {target_cities}
    - Province: {province}
    - Country: {country}

    ORIGINAL DRAFT:
    {draft}

    TASK:

    1. Do NOT modify or rewrite the original content text.

    2. For each of the {draft_option_no} options, generate:
       - SEO Meta Title (max 60 characters)
       - Meta Description (max 160 characters)

    3. Organize the content for the {draft_option_no} options using proper markdown structure:
       - H1 for main title
       - H2 for sections
       - H3 if necessary for sub-sections

    4. Ensure each of the {draft_option_no} options is clearly separated
       and easy to distinguish.

    5. Maintain original tone and brand voice exactly as written.

    OUTPUT:
    Return fully structured SEO-optimized markdown content.
    """

    user_prompt = user_prompt_template.format(
        draft_option_no=draft_option_no,
        b_type=b_type,
        business_name=business_name,
        target_cities=target_cities,
        province=province,
        country=country,
        draft=draft
    )


    # -----------------------------
    # 5. Invoke SEO agent
    # -----------------------------
    response = seo_agent.invoke({
        "input": user_prompt
    })

    # -----------------------------
    # 6. Return updated state
    # -----------------------------
    return {
        # SEO-enhanced final output
        "landing_page_draft": response.content
    }
