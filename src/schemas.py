from typing import Annotated, TypedDict, List
import operator
from dataclasses import dataclass


@dataclass
class BusinessDetail(TypedDict):
    """
    Represents a business detail data classwith name, URL, summary, and city.

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