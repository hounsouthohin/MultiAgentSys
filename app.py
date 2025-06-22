from smolagents import tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from smolagents.cli import load_model


from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, OpenAIServerModel
from smolagents.cli import load_model

from Functions.parse_arguments import parse_arguments
from Functions.save_screenshot import save_screenshot
from Functions.initialized_driver import initialize_driver   

import yaml 
from dotenv import load_dotenv
import time 




################################################################################## TOOLSfrom selenium.webdriver.common.by import By
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Scrolls to the nth paragraph that contains the given text and returns its content.

    Args:
        text (str): Text to search for in paragraphs (case-insensitive).
        nth_result (int): The nth occurrence to return (1-based index).

    Returns:
        str: The content of the matched paragraph.
    """
    # Recherche insensible à la casse dans tous les paragraphes
    paragraphs = driver.find_elements(By.XPATH, f"//p[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]")

    if not paragraphs:
        raise Exception(f"Aucun paragraphe contenant '{text}' n’a été trouvé.")
    if nth_result > len(paragraphs):
        raise Exception(f"'{text}' trouvé {len(paragraphs)} fois, mais la {nth_result}e occurrence a été demandée.")

    target_paragraph = paragraphs[nth_result - 1]

    # Scroll jusqu’au paragraphe
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", target_paragraph)
    time.sleep(2)

    content = target_paragraph.text.strip()
    print(f"🔍 Paragraphe trouvé (#{nth_result}) contenant '{text}':\n{content}")
    return content



# Add a function to save Screenshots 


@tool
def go_back() -> None:
    """Goes back to previous page."""
    # Go back
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    driver = webdriver.Chrome()
    # send escape key used to simulate a user pressing the escape key
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    
    

# On instancie l'outil ici
duck_search = DuckDuckGoSearchTool()

@tool
def web_search(query: str) -> str:
    """A tool that searches the web using DuckDuckGo.
    Args:
        query: A string representing the search query.
    """
    # On appelle l'objet comme une fonction
    return duck_search(query)


import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

llava_pipe = pipeline(
    "image-to-text",
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    device=device
)



from smolagents import AgentMemory, ActionStep


@tool
def describe_last_screenshot() -> str:
    """
    Use the visual model to describe the last screenshot taken by the agent.

    Returns:
        str: A description of the last screenshot based on the visual model.
    """
    global AGENT_MEMORY
    if AGENT_MEMORY is None:
        return "Agent memory is not available."

    for step in reversed(AGENT_MEMORY.steps):
        if isinstance(step, ActionStep):
            if hasattr(step, "observations_images") and step.observations_images:
                img = step.observations_images[-1]
                description = llava_pipe(img)[0]["generated_text"]
                return f"🖼 Step {step.step_number} screenshot description:\n{description}"
    return "No screenshot found in memory."



################################################################################## FUNCTIONS
def initialize_agent(model):
    
    """Initialize the CodeAgent with the specified model."""
    
    return CodeAgent(
        tools=[go_back, close_popups, search_item_ctrl_f, web_search,describe_last_screenshot],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )
    




    
    
################################################################################## MAIN"


def main():
    global AGENT_MEMORY  # pour que la mémoire soit accessible à l'outil
    # Load environment variables
    load_dotenv()

    # Load system prompt from prompt.yaml file
    with open("prompts.yaml", 'r') as stream:
        prompt = yaml.safe_load(stream)
    
    # Parse command line arguments
    args = parse_arguments()

    # Initialize the model based on the provided arguments
    model = load_model(args.model_type, args.model_id)

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)
    
    AGENT_MEMORY = agent.memory  # <- on conserve la mémoire pour l'outil

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(args.prompt + prompt["helium_instructions"])
if __name__ == "__main__":
    main()
