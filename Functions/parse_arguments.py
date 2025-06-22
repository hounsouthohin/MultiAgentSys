import argparse

alfred_guest_list_request = """
    I am Alfred, the butler of Wayne Manor, responsible for verifying the identity of guests at party. A superhero has arrived at the entrance claiming to be Wonder Woman, but I need to confirm if she is who she says she is.
    Please search for images of Wonder Woman and generate a detailed visual description based on those images. Additionally, navigate to Wikipedia to gather key details about her appearance. With this information, I can determine whether to grant her access to the event.
    """
parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    
def parse_arguments():
    
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=alfred_guest_list_request, # Default prompt
        help="The prompt to run with the agent",
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel,ollama.0.9.2)",
    )   
    parser.add_argument(
        "--model-id",
        type=str,
        default="ollama.0.9.2",
        help="The model ID to use for the specified model type",
    )
    return parser.parse_args()
