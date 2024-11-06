### UTILTIY FUNCTIONS

import os
import json
import base64
from typing import Literal, Union

from sray_ValidatedLLM.modules.constants import DataConstants

import openai
from openai import AzureOpenAI, OpenAI, AsyncOpenAI

### COLORS

def fmt(text, n="DEFAULT", just_code=False, end=""):
    TERMINATE_SEQ = "\033[0m"
    NO_CHANGE = ""
    
    assert isinstance(text, str), "Input text should be a string."
    assert isinstance(n, str), "Color format should be a string."
    assert (n == "TERMINATE" and just_code) or (n != "TERMINATE"), "If `n` is `TERMINATE`, then `just_code` should be True."
    assert (just_code and text == "") or (not just_code), "If `just_code` is True, then text should be empty."
    assert (just_code) or (end == "" and not just_code), "It only makes sense to specify `end` if `just_code` is True."
    
    mapping = {"red": "\033[91m",
               "green": "\033[92m",
               "blue": "\033[94m",
               "yellow": "\033[93m",
               "purple": "\033[95m",
               "cyan": "\033[96m",
               "black": "\033[90m",
               "bold": "\033[1m",
               "underline": "\033[4m",
               "italic": "\033[3m",
               "gray": "\033[97m",
               "TERMINATE": TERMINATE_SEQ}
    assert "DEFAULT" not in mapping, "`DEFAULT` is a reserved keyword for this function."
    assert n in mapping or n == "DEFAULT", f"Color `{n}` not supported. If you want default color, use `DEFAULT`/omit argument specification."
    
    if just_code:
        print(mapping.get(n, NO_CHANGE), end=end)
    else:
        return mapping.get(n, NO_CHANGE) + str(text) + TERMINATE_SEQ

### File Reading/Writing

def path_exists(path: str) -> bool:
    return os.path.exists(path)

def path_is_file(path: str) -> bool:
    return os.path.isfile(path)

def load_json(load_path: str) -> dict:
    with open(load_path, "r") as f:
        return json.load(f)

### Data Wrangling Utilities

def wrap_for_unpacking(thing) -> tuple:
    if thing is None:
        return ()
    else:
        return (thing,)

### OpenAI Client Setup

def get_credentials(mid_req_by_user: str) -> dict:
    def _check_environment(model_id_requested_by_user: str):
        # Initialize the result dictionary
        result = {}
        
        # Check and add the required variables to the result dictionary
        for var in DataConstants.CREDENTIALS_VARS_REQUIRED:
            value = os.getenv(var)
            if not value:
                raise EnvironmentError(f"Required env-var '{var}' is missing.")
            result[var] = value
        
        # Check and add at least one of the optional variables to the result dictionary
        any_one_of_found = False
        model_id_matched = False
        for var in DataConstants.CREDENTIALS_VARS_ANY_ONE_OF:
            value = os.getenv(var)
            if value:
                result[var] = value
                any_one_of_found = True
                if var == model_id_requested_by_user:
                    model_id_matched = True
        
        if not any_one_of_found:
            raise EnvironmentError(f"At least one of the optional environment variables '{', '.join(DataConstants.CREDENTIALS_VARS_ANY_ONE_OF)}' must be set.")
        if not model_id_matched:
            raise EnvironmentError(f"The model ID requested by the user '{model_id_requested_by_user}' is not set in the environment.")
        
        if len(result) < len(DataConstants.CREDENTIALS_VARS_REQUIRED) + 1:
            raise EnvironmentError(f"Expected {len(DataConstants.CREDENTIALS_VARS_REQUIRED) + 1} environment variables, got {len(result)}.")
        
        return result
    
    def _check_configfile(model_id_requested_by_user: str):
        credentials = load_json(DataConstants.CREDENTIALS_PATH)
        # Check that all the required keys are present
        for var in DataConstants.CREDENTIALS_VARS_REQUIRED:
            if var not in credentials:
                raise Exception(f"Required key '{var}' not found in credentials file.")
        
        # Check that the requested model ID is present
        if model_id_requested_by_user not in credentials:
            raise Exception(f"Requested model ID '{model_id_requested_by_user}' not found in credentials file.")
        
        return credentials
    
    EXPECTED_FORMAT = """
{
    "AZURE_OPENAI_API_KEY": "SOME KEY (32 characters long)",
    "AZURE_OPENAI_ENDPOINT": "SOME ENDPOINT URL (eg. "https://cicero-dev-open-ai.openai.azure.com")",
    "AZURE_OPENAI_API_VERSION": "SOME API VERSION (eg. 2024-04-01-preview)",
    "AZURE_GPT4o_MODEL_ID": "cicero-dev-gpt-o",
    "AZURE_GPT4_TURBO_MODEL_ID": "cicero-dev-gpt-4"
}
"""
    
    credentials = None
    if not path_exists(DataConstants.CREDENTIALS_PATH):
        # If there's no credentials file, then check if the environment has the required variables
        try:
            credentials = _check_environment(mid_req_by_user)
        except EnvironmentError as e:
            # If the environment does not have the required variables either, then raise an exception
            _msg = fmt(f"\nCredentials file {DataConstants.CREDENTIALS_PATH} does not exist or not stored as environment variables. Expected format: {EXPECTED_FORMAT}or store these key-values as environment variables.", "red")
            raise Exception(_msg)
    else:
        # If there is a credentials file, then just load it in.
        credentials = _check_configfile(mid_req_by_user)
    assert credentials is not None, "FATAL: Credentials supposedly exist, but could not be loaded."
    
    return credentials

def configure_openai(model_id_to_use: str, client_host: Literal["openai", "azureopenai", "openai-async"]) -> tuple[Union[AzureOpenAI, OpenAI], str]:
    """Configure the OpenAI client and retrieve the model ID.

    Raises:
        Exception: If any of the required credentials are missing.

    Returns:
        tuple[openai.lib.azure.AzureOpenAI / OpenAI, str]: The configured OpenAI client
            and the model ID.
    """
    assert client_host in ["openai", "azureopenai", "openai-async"], f"Client host {client_host} is not supported. Supported client hosts are ['openai', 'azureopenai', 'openai-async']."
    # assert model_id_to_use in DataConstants.DFT_MODEL_IDS_SUPPORTED, f"Model ID {model_id_to_use} is not supported. Supported model IDs are {DataConstants.DFT_MODEL_IDS_SUPPORTED}."
    
    _credentials = None
    if client_host == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_id = os.environ["DEV_OPENAI_MODEL_ID"]
    elif client_host == "openai-async":
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_id = os.environ["DEV_OPENAI_MODEL_ID"]
    elif client_host == "azureopenai":
        _credentials = get_credentials(mid_req_by_user=model_id_to_use)
        try:
            model_id = _credentials[model_id_to_use]
            client = AzureOpenAI(api_key=_credentials['AZURE_OPENAI_API_KEY'],  
                                api_version=_credentials['AZURE_OPENAI_API_VERSION'],
                                azure_endpoint=_credentials['AZURE_OPENAI_ENDPOINT'])
        except KeyError:
            raise Exception("Please check your credentials file. Should have all keys requested in this function.")
    
    return client, model_id

### Pre-LLM Call Utilities

def encode_image(image_path: str) -> str:
    assert path_exists(image_path), f"Image path {image_path} does not exist."
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")