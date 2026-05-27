import streamlit as st


def compute_progress(
    no_of_cities,
    current_city_index,
    designer_progress,
    editor_progress,
    revision_count,
    max_revisions,
    seo_progress
) -> float:

    """
    Computes overall workflow progress for a multi-stage LangGraph pipeline.

    The progress score is a weighted combination of different pipeline stages:
    - Search progress (based on city iteration)
    - Designer stage progress
    - Editor stage progress
    - Revision progress (based on revision count vs max revisions)
    - SEO finalization progress

    Each component contributes a fixed portion of the total 100% scale:
    - Search progress: scaled across cities (0-40%)
    - Designer progress: external stage contribution (0-10 %)
    - Editor progress: external stage contribution (0-10 %)
    - Revision progress: scaled by revision completion (0-30%)
    - SEO progress: finalization stage contribution (0-10 %)

    Args:
        no_of_cities (int):
            Total number of cities being processed in the workflow.

        current_city_index (int):
            Index of the current city being processed (0-based).

        designer_progress (float):
            Progress contribution from the content generation (designer) stage.

        editor_progress (float):
            Progress contribution from the editorial review stage.

        revision_count (int):
            Number of completed revision cycles for the current draft.

        max_revisions (int):
            Maximum allowed revision cycles.

        seo_progress (float):
            Progress contribution from the SEO optimization stage.

    Returns:
        float:
            Total computed progress percentage, clamped between 0 and 100.
    """

    search_progress = (
        ((current_city_index) / max(no_of_cities, 1)) * 40
    )

    revision_progress = (
        (revision_count / max(max_revisions, 1)) * 30
    )

    return min(
        100,
        search_progress + designer_progress + editor_progress + revision_progress + seo_progress
    )


def run_workflow(
    graph,
    query: str,
    business_name: str,
    business_type: str,
    cities: list,
    target_cities: str,
    province: str,
    country: str,
    max_sites: int,
    draft_option_no: int
):

    """
    Executes the LangGraph workflow for generating website content from search to SEO optimization.

    The function initializes the full agent state and triggers the graph execution.

    Workflow includes:
    - Multi-city search iteration
    - Content generation (designer node)
    - Editorial review and revision loop
    - SEO optimization finalization

    Args:
        graph: Compiled LangGraph instance used for execution.
        query (str): Search query used to find competitor/business data.
        business_name (str): Name of the business being generated.
        business_type (str): Industry or category of the business.
        cities (list): List of cities to iterate through during search.
        target_cities (str): Target geographic market description.
        province (str): Province or state context.
        country (str): Country context.
        max_sites (int): Maximum number of websites to retrieve per city.
        draft_option_no (int): Number of landing page draft variations to generate.

    Returns:
        dict: Final state output from the LangGraph execution.
    """

    # -------------------------------------------------
    # Initialize full graph input state
    # -------------------------------------------------
    initial_inputs = {
        "query": query,
        "business_name": business_name,
        "business_type": business_type,
        "cities": cities,
        "current_city_index": 0,
        "no_of_cities": len(cities),
        "current_city": cities[0],
        "target_cities": target_cities,
        "province": province,
        "country": country,
        "max_sites": max_sites,

        # runtime / workflow state
        "search_results": [],
        "landing_page_draft": "",
        "draft_option_no": draft_option_no,
        "last_critique": "",
        "revision_count": 0,
        "max_revisions": 3,
        "approval_status": "REJECTED"
    }

    # variables used for progress status bar
    progress_bar = st.progress(0)
    status = st.empty()
    no_of_cities = 1
    current_city_index = 0
    current_city = cities[0]
    designer_progress = 0
    editor_progress = 0
    revision_count = 0
    max_revisions = 3
    seo_progress = 0

    progress = compute_progress(
        no_of_cities,
        current_city_index,
        designer_progress,
        editor_progress,
        revision_count,
        max_revisions,
        seo_progress
    )
    progress_bar.progress(int(progress))
    status.text(f"{int(progress)}%")

    # variables used for graph streaming
    cnt = 0
    current_state = {}
    
    # Stream the graph execution output
    for output in graph.stream(initial_inputs):
        # 'output' is a dictionary where the key is the name of the node that just ran
        for node_name, state_update in output.items():

            if cnt == 0:
                st.markdown("Starting Website Content Generation...")
                st.markdown("")
                cnt += 1

            if (node_name == "search") and ("search_results" in state_update):
                # Update search progress variables
                no_of_cities = int(state_update['no_of_cities'])
                current_city_index = int(state_update['current_city_index'])

                current_city = state_update['current_city']

                st.markdown(f"🔍 Retrieving Website Content Search Results for {current_city}...")
                st.markdown("---")

            elif (node_name == "designer") and ("landing_page_draft" in state_update):
                # Set designer progress to 10% once the first draft is created
                designer_progress = 10
                st.markdown("🎨 First draft has been created...")
                st.markdown("---")
            
            elif (node_name == "editor") and ("approval_status" in state_update):
                # Set editor progress to 10% when editing
                editor_progress = 10
                # revision_count is variable, so we set it to max_revisions for progress bar update
                if revision_count < max_revisions:
                    revision_count = max_revisions
      
                st.markdown("✅ The draft has been reviewed...")
                st.markdown("---")

            elif node_name == "revision":
                # Merge the latest update from the current LangGraph node into the running application state.
                # This ensures that any new or modified keys from `state_update` are added to `current_state`
                current_state.update(state_update)

                # Update revision progress variables
                revision_count = current_state.get("revision_count", 0)
                max_revisions = current_state.get("max_revisions", max_revisions)

                st.markdown(f"🔁 The draft has been revised...")
                st.markdown("---")

            elif node_name == "seo":
                if state_update.get("landing_page_draft"):
                    # Set SEO progress to 10% at completion of content generation
                    seo_progress = 10

                    progress = compute_progress(
                        no_of_cities,
                        current_city_index,
                        designer_progress,
                        editor_progress,
                        revision_count,
                        max_revisions,
                        seo_progress
                    )

                    progress_bar.progress(int(progress))
                    status.text(f"{int(progress)}%")

                    st.markdown("✅ Website content generation complete!")
                    st.markdown("")
                    st.markdown(f"Content:\n\n{state_update['landing_page_draft']}")
                    st.markdown("---")

            progress = compute_progress(
                no_of_cities,
                current_city_index,
                designer_progress,
                editor_progress,
                revision_count,
                max_revisions,
                seo_progress
            )

            progress_bar.progress(int(progress))
            status.text(f"{int(progress)}%")



