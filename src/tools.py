
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import tool
from src.config import get_api_key
import os

# The function `filter_data` is a helper function that is used by the search_website_content function to filter out websites, using keywords, that are not directly business domains.
def filter_data(raw_data, max_sites):

    exclude_keywords = ["top 10", "best 10", "10 best", "mapquest", "10 professional", "companies in", "services in", "yelp", "threebestrated", "houzz", "yellow pages", "facebook", "instagram", "linkedin", "twitter", "pinterest", "tiktok", "reddit", "quora", "youtube", "blog", "news", "press", "media", "press release", "press releases", "press coverage", "press coverage of", "press coverage of ", "press coverage of the", "press coverage of the ", "press coverage of the company", "press coverage of the company ", "press coverage of the company's", "press coverage of the company's ", "press coverage of the company's news", "press coverage of the company's news ", "press coverage of the company's news and updates", "press coverage of the company's news and updates ", "press coverage of the company's news and updates and more", "press coverage of the company's news and updates and more"]
    filtered_results = []
    seen_urls = set()

    for item in raw_data:
        title = item.get('title', '').lower()
        url = item.get('url', '').split('?')[0] # Clean tracking IDs

        # 1. Filter by keyword
        if any(k in title for k in exclude_keywords):
            continue
        
        # 2. De-duplicate
        if url in seen_urls:
            continue
        
        # 3. Add to final list
        seen_urls.add(url)
        filtered_results.append({
            "name": item.get('title'),
            "url": url,
            "summary": item.get('content') # Trim content to keep context light
        })

        # Stop once we hit the user's requested limit
        if len(filtered_results) >= max_sites:
            break

    # Return as a string so the LLM can easily "read" it
    return str(filtered_results)

# Define a tool function called `search_website_content` that will search for market competitor website content. 
# Langchain has a tool definition function: `tool`. We can define the tool function as a Tool using `@tool` decorator. 
# LangChain also has a built-in tool for this: `TavilySearchAPIWrapper` that can be used for current targetted web searches. 
# The function will take in the business type, city, province, country, and max sites as parameters along with the actual query 
# that can be passed to the TavilySearchAPIWrapper. 
# TavilySearchAPIWrapper also uses configuartion parameters to set the max results, search depth, and other settings. 
# The search results are returned as a list of dictionaries.
@tool
def search_website_content(query: str, business_type: str, city: str, province: str, country: str, max_sites: int = 5) -> list:
    """
    Searches for market competitor website content. 
    Args:
        query: The specific search query.
        business_type: The industry or type of business.
        city: The target city.
        province: The target province/state.
        country: The target country.
        max_sites: Maximum number of sites to return (default: 5)
    """

    os.environ["TAVILY_API_KEY"] = get_api_key("TAVILY_API_KEY")
    
    full_query = f"{business_type} in {city}, {province}, {country}: {query}"
    
    tavily_search = TavilySearchAPIWrapper()

    RESEARCH_MODE_CONFIG = {
        "max_results": max_sites,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": True,
        "include_images": False
    } 

    raw_data = tavily_search.results(query=full_query, **RESEARCH_MODE_CONFIG)
    
    filtered_results = filter_data(raw_data, max_sites)

    return filtered_results