
from src.create_agent import create_langchain_llm
from src.run_workflow import run_workflow
from src.build_graph import build_graph
import re
import json
import streamlit as st

# This app uses a LangGraph workflow to generate website content and SEO optimization for a business.
# We will have 4 agents in the workflow:

# 1. Search Agent: Finds relevant websites for the business by business type for a list of locations using the Tavily Search tool.
# 2. Designer Agent: Creates initial website landing page content based on research
# 3. Editor Agent: Reviews and approves the content
# 4. Revision Agent: Revises the content based on the editor's feedback
# 5. SEO Agent: Optimizes the content for search engines as the final step

# The following input variables are used in the workflow:
    # query (str): Search query used to find competitor/business data.
    # business_name (str): Name of the business being generated.
    # business_type (str): Industry or category of the business.
    # cities (list): List of cities to iterate through during search.
    # target_cities (str): Target geographic market description.
    # province (str): Province or state context.
    # country (str): Country context.
    # max_sites (int): Maximum number of websites to retrieve per city.
    # draft_option_no (int): Number of landing page draft variations to generate.

# The main page does the following:
    # 1. Authenticates the user in the sidebar before using the app and displaying the other UI features
    # 2. Displays the model selection in the sidebar to choose an LLM model
    # 3. Collects business input details in a form in the sidebar
    # 4. Validates the form inputs before continuing
    # 5. Runs the LangGraph workflow
    # 6. Displays the results in the main page


def app_authentication():
    """
    Handles simple password-based authentication for the Streamlit app.

    This function gates access to the application using a password stored
    in Streamlit secrets. It prevents the rest of the app from executing
    until the user successfully authenticates.

    Behavior:
        - Displays a password input field if the user is not authenticated
        - Compares user input against the stored secret password
        - Sets session state authentication flag upon success
        - Stops app execution for unauthenticated users using st.stop()

    Security Notes:
        - This is a lightweight access control mechanism intended for
          preventing unauthorized usage, not for high-security authentication.
        - The password is stored in Streamlit secrets (APP_PWD).

    Side Effects:
        - Updates st.session_state.authenticated
        - May trigger st.rerun() after successful login
        - Halts execution for unauthenticated users
    """

    pwd = st.secrets.get("APP_PWD")

    if not st.session_state.authenticated:
        
        st.header("Authentication")

        psswd = st.text_input("Enter password", type="password")

        if st.button("Login"):
            if psswd == pwd:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        
        st.stop()

def load_models(json_path: str) -> dict:
    """
    Parses a JSON configuration file to retrieve AI model metadata.

    Instead of raising exceptions, this function catches them and displays 
    user-friendly error messages in the Streamlit UI before halting execution.

    Args:
        json_path (str): The file path to the models.json configuration file.

    Returns:
        dict: A nested dictionary of model details if successful.
    
    Error Handling:
        - On FileNotFoundError: Displays error and stops the app.
        - On JSONDecodeError: Displays syntax error details and stops the app.
        - On general Exception: Displays the traceback and stops the app.
    """
    try:
        # Use utf-8 to ensure special characters in descriptions load correctly
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
          
    except FileNotFoundError:
        st.error(f"🚨 **Configuration Missing:** The file '{json_path}' was not found.")
        st.stop() 

    except json.JSONDecodeError as e:
        st.error(f"📑 **Syntax Error:** The file '{json_path}' contains invalid JSON.")
        st.info(f"Check line {e.lineno}, column {e.colno}.")
        st.stop() 

    except Exception as e:
        st.error(f"❌ **Unexpected Error:** An issue occurred while loading '{json_path}'.")
        st.exception(e) # Collapsible technical details for debugging
        st.stop()


def get_models_for_provider(models_data: dict, provider: str) -> dict:
    """
    Extracts the subset of models belonging to a specific AI provider.

    This helper function filters the global models dictionary to return only 
    the models associated with the user's selected provider (e.g., 'Google'). 
    It uses a fallback mechanism to prevent the UI from breaking if a 
    provider key is missing.

    Args:
        models_data (dict): The full dictionary loaded from models.json.
        provider (str): The name of the provider to look up (e.g., "OpenAI").

    Returns:
        dict: A dictionary of models for that provider, or an empty 
              dictionary {} if the provider is not found.
    """
    # Using .get() instead of models_data[provider] is a "defensive" move.
    # If 'provider' isn't a key in the dictionary, it returns the second 
    # argument ({}) instead of raising a KeyError.
    return models_data.get(provider, {})


def render_model_description(info: dict, provider: str) -> None:
    """
    Renders a formatted UI block showing model metadata and pricing.

    This function takes a dictionary of model information and displays it 
    using a mix of Streamlit's native components and raw HTML/CSS. It 
    provides the user with the model's originating company, a brief 
    functional description, and the cost per million tokens.

    Args:
        info (dict): The specific model dictionary from models.json 
                     (e.g., {"Company": "Google", "Input": 3.50, ...}).
        provider (str): The fallback provider name if 'Company' is missing.

    Returns:
        None
    """

    # 1. DATA EXTRACTION: Pull the pricing or default to "N/A" to prevent errors.
    input_cost  = info.get("Input",  "N/A")
    output_cost = info.get("Output", "N/A")

    # 2. HEADER: Displays the building emoji and the company name in bold.
    st.caption(f"🏢 **{info.get('Company', provider)}**")

    # 3. BODY: Displays the model's intended use case or description.
    st.write(info.get("Description", "No description available."))

    # 4. FOOTER (HTML): Creates a small, grey, "pro-style" pricing line.
    # Uses HTML entities: &#128176; (Money Bag) and &#36; (Dollar Sign).
    st.markdown(
        f"<p style='color:grey;font-size:0.85em;margin:0'>"
        f"&#128176; Input: &#36;{input_cost} / 1M tokens"
        f" &nbsp;&middot;&nbsp; "
        f"Output: &#36;{output_cost} / 1M tokens"
        f"</p>",
        unsafe_allow_html=True,
    )

def model_selector(models_data: dict, providers: list) -> tuple:
    """
    Renders a Streamlit UI component for selecting an LLM provider and model.

    Args:
        models_data (dict): Nested dictionary containing provider and model metadata.

    Returns:
        tuple: (provider_name, model_name) or (provider_name, None) if no model selected.
    """

    with st.expander("Model Selection", expanded=True):
        #with st.container(border=True):
        # 1. Initialize Provider Selection
        provider = st.selectbox(
            "Choose the LLM Provider", 
                providers, 
                key="provider"
        )

        # 2. Fetch models for the chosen provider and filter exclusions
        provider_models = get_models_for_provider(models_data, provider)
        all_names = list(provider_models.keys())

        options = ["Select Model..."] + [m for m in all_names]

        # 3. Render Model Selection
        model = st.selectbox(
            "LLM", 
            options, 
            key="model"
        )

        # 4. Implement a slider for LLM temperature for Designer agent with a default value of 0.3 and step of 0.1
        temperature = st.slider(
            "Temperature (Creativity Level)",
            0.0,
            1.0,
            0.3,
            step=0.1
        )

        # 5. UI Feedback: Show model details if a valid selection is made
        if model and model != "Select Model...":
            # Pull metadata from the dictionary subset
            model_info = provider_models.get(model, {})
            render_model_description(model_info, provider)

            return provider, model, temperature
        else:
            st.stop()

    return provider, None


def initialize_inputs():
    """
    Initializes default Streamlit session state variables used by the application.

    This function ensures that required session state keys exist before the UI
    or workflow logic is executed. Default values are only assigned if the
    session state variable has not already been created, preventing user inputs
    from being overwritten during Streamlit reruns.

    Initialized session state variables:

        authenticated (bool):
            Whether the user is authenticated to use the application.

        query (str):
            Search query used for website content retrieval.

        max_sites (int):
            Maximum number of competitor websites to retrieve per search.

        draft_option_no (int):
            Number of landing page draft variations to generate.

        missing_fields (list):
            Collection of missing input field names used for validation feedback and error dialogs.

        PROVIDERS (list):
            List of available AI model providers.

    Returns:
        None
    """

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "max_sites" not in st.session_state:
        st.session_state.max_sites = 3
    #if "draft_option_no" not in st.session_state:
        #st.session_state.draft_option_no = 2
    if "missing_fields" not in st.session_state:
        st.session_state.missing_fields = []
    if "providers" not in st.session_state:
        st.session_state["providers"] = ["OpenAI", "Google", "Groq", "HuggingFace"]


def validate_missing_inputs():
    """
    Validates required Streamlit session state input fields.

    The function checks whether required user input fields stored in
    Streamlit session state contain values. Any missing or empty fields
    are collected and returned as a list for validation feedback and
    error dialog display.

    Required fields validated:
        - cities
        - target_cities
        - business_type
        - business_name
        - country
        - province

    Returns:
        list[str]:
            A list containing the names of missing or empty input fields.
            Returns an empty list if all required fields are populated.
    """

    errors = []

    if not st.session_state.cities:
        errors.append("Cities")

    if not st.session_state.target_cities:
        errors.append("Target Cities")

    if not st.session_state.business_type:
        errors.append("Business Type")

    if not st.session_state.business_name:
        errors.append("Business Name")

    if not st.session_state.country:
        errors.append("Country")

    if not st.session_state.province:
        errors.append("Province/State")

    return errors


@st.dialog("Missing Required Fields")
def error_missing_fields_dialog():
    """
    Displays a modal dialog containing a list of missing input fields.

    The dialog is triggered when form validation fails and uses the
    `missing_fields` values stored in Streamlit session state to dynamically
    render validation feedback to the user.

    Behavior:
        - Displays all missing fields as a bullet-style list.
        - Allows the user to acknowledge the validation message using the
          "OK" button.
        - Resets validation-related session state variables when dismissed.
        - Forces a Streamlit rerun to refresh the UI state cleanly.

    Session State Variables Used:
        missing_fields (list):
            List of field names that failed validation.

        show_missing_error (bool):
            Flag controlling whether the dialog should be displayed.

    Returns:
        None
    """

    st.write("Please complete the following fields:")

    for field in st.session_state.missing_fields:
        st.write(f"- {field}")

    if st.button("OK"):
        st.session_state.missing_fields = []
        st.rerun()


def validate_city_input(text: str) -> bool:
    """
    Validates and normalizes a city input string.

    The function:
        - Converts multiple spaces into commas
        - Ensures consistent comma-separated formatting
        - Validates that no empty city entries exist

    Args:
        text (str):
            Raw user input containing cities separated by commas or spaces.

    Returns:
        bool:
            True if the input is valid after normalization, otherwise False.
    """

    if not text:
        return False

    # 1. normalize: convert any whitespace runs into commas
    normalized = re.sub(r"\s+", ",", text.strip())

    # 2. split and clean
    parts = [p.strip() for p in normalized.split(",") if p.strip()]

    # 3. validate
    return len(parts) > 0


@st.dialog("Cities entered are invalid")
def error_invalid_field_dialog(field_name: str):
    """
    Displays a Streamlit modal dialog when a city input field fails validation.

    This dialog informs the user that the provided city input is invalid and
    must contain at least one valid city or multiple cities separated by commas.

    The dialog is parameterized by the field name so it can be reused for
    different city-related inputs (e.g., 'Cities' or 'Target Cities').

    Behavior:
        - Displays a validation message explaining the expected input format.
        - Shows an acknowledgment button ("OK") to dismiss the dialog.
        - Resets the validation error flag in Streamlit session state.
        - Forces a Streamlit rerun to refresh UI state after dismissal.

    Args:
        field_name (str):
            The name of the input field being validated (used for context in
            the error message).

    Session State Variables Used:
        show_validation_error (bool):
            Controls whether the validation error dialog is displayed.

    Returns:
        None
    """

    st.write(f"Required: at least one city or multiple cities separated by commas for {field_name}.")

    if st.button("OK"):
        st.rerun()


def parse_cities(text: str) -> list[str]:
    """
    Converts a comma-separated city string into a cleaned list of city names.

    The function:
        - Splits the input string using commas.
        - Removes leading and trailing whitespace from each city name.
        - Excludes empty or whitespace-only entries.

    Args:
        text (str):
            Raw comma-separated city input string.

    Returns:
        list[str]:
            Cleaned list of city names.
    """

    return [c.strip() for c in text.split(",") if c.strip()]
    

# Set page configuration
st.set_page_config(
    page_title="🤖 AI Website Content Generation and Search Engine Optimization (SEO)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load initial variables
initialize_inputs()

# Set columns for header sizing
col1, col2, col3 = st.columns([1, 15, 1])

with col2:
    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <h2 style="margin: 0;">🤖 AI Website Content Generation and Search Engine Optimization (SEO)</h2>
        </div>
    """, unsafe_allow_html=True)

# Set columns for description and content generation box
col1, col2, col3 = st.columns([1, 8, 1])

description = """
    This Streamlit application generates website content using multiple AI agents.
    Once the application is authenticated, the user can select the LLM model to use for content generation.
    Draft option selection allows the user to specify the number of content drafts to generate.

    The application uses the following coding frameworks and tools:
    - Tavily API for competitor research
    - LangChain for LLM integration
    - LangGraph for workflow management

    The workflow uses the selected LLM and includes the following steps in sequence:
    - competitor research
    - content generation
    - editorial review (can provide feedback up to 3 times)
    - content revision (based on editorial feedback)
    - SEO analysis and optimization
"""

with col2:
    # Description box
    st.markdown(
        f"""
        <div style="
            border: 0px;
            border-radius: 10px;
            padding: 10px;
        ">
        {description}
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:

    # Authenticate app before moving forward
    app_authentication()

    # Load model options
    models_data = load_models("models.json")

    st.subheader("Large Language Model (LLM) Selection")
    st.write("Choose your AI model for the workflow. Then adjust the temperature (creativity level) for the initial draft generation.")

    # Get provider, model name and selected temperature from selector
    llm_provider, model_name, temperature2 = model_selector(models_data, st.session_state["providers"])

    # Set default LLM temperature of all agents except Designer
    temperature1 = 0.3
    llm1 = create_langchain_llm(llm_provider=llm_provider, model_name=model_name, temperature=temperature1)
    
    # Get LLM for Designer agent
    llm2 = create_langchain_llm(llm_provider=llm_provider, model_name=model_name, temperature=temperature2)
    
    st.divider()

    st.subheader("Business Details Selection")
    st.write("Fill in the business details below to help with content generation. Select the no. of options generated for the final draft.")

    # Input form for business details
    with st.form("input_form"):

        st.text_input("Business Type", key="business_type")

        if st.session_state.business_type != "":
            st.session_state.query=f"Retrieve website content for {st.session_state.business_type} businesses."

        st.text_input("Business Name", key="business_name")
        st.text_input("Country", key="country")
        st.text_input("Province/State", key="province")
        st.text_input("Cities", key="cities")
        st.text_input("Target Cities", key="target_cities")
        
        # Implement a slider for number of draft options with a default value of 1 and step of 1
        st.slider(
        "Draft Options",
        1,
        10,
        1,
        step=1,
        key="draft_option_no"
        )

        # Submit button
        submitted = st.form_submit_button("Submit", use_container_width=True)


with col2:
    # Content generation box
    with st.container(height=800, border=True):

        # If submit button is clicked
        if submitted:

            # -------------------------------------------------
            # 1. Validate missing or invalid inputs
            # -------------------------------------------------
            missing_errors = validate_missing_inputs()
            city_invalid = not validate_city_input(st.session_state.cities)
            target_city_invalid = not validate_city_input(st.session_state.target_cities)

            # -------------------------------------------------
            # 2. If missing errors or invalid inputs, show dialogues
            # -------------------------------------------------
            if missing_errors:
                st.session_state.missing_fields = missing_errors
                error_missing_fields_dialog()
                st.stop()
            elif city_invalid:
                error_invalid_field_dialog("Cities")
                st.stop()
            elif target_city_invalid:
                error_invalid_field_dialog("Target Cities")
                st.stop()
            
            # -------------------------------------------------
            # 3. If valid inputs → normalize + run graph
            # -------------------------------------------------

            # structured list for graph
            cities = parse_cities(st.session_state.cities)

            # string for LLM / context
            target_cities = ", ".join(parse_cities(st.session_state.target_cities))

            # optional derived query
            st.session_state.query = f"Retrieve website content for {st.session_state.business_type} businesses."
            
            # Build the graph
            graph = build_graph(llm1, llm2)

            # Run the workflow with initial inputs
            run_workflow(
                graph,
                st.session_state.query,
                st.session_state.business_name,
                st.session_state.business_type,
                cities,
                target_cities,
                st.session_state.province,
                st.session_state.country,
                st.session_state.max_sites,
                st.session_state.draft_option_no
            )

