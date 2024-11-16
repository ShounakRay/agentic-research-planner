from modules.constants import DataConstants
from modules.llm_funcs import prompt_LLM, show_io
from modules.utilities import configure_openai, encode_image, fmt, path_exists

import openai

# FIXME: Remove this line if you don't want to see the input/output of the LLM
# when you call the `prompt_LLM` function.
prompt_LLM = show_io(prompt_LLM)

"""


"""
def _meta_check(CHOSEN_MODEL_ID: str, PATH_TO_LLM_IMAGE: str):
    assert CHOSEN_MODEL_ID in DataConstants.DFT_MODEL_IDS_SUPPORTED, f"Chosen model ID is not in the list of possible model IDs, {DataConstants.DFT_MODEL_IDS_SUPPORTED}."
    if PATH_TO_LLM_IMAGE is not None:
        assert path_exists(PATH_TO_LLM_IMAGE), "Path to image does not exist. If you don't want an image, set PATH_TO_LLM_IMAGE to None."
        assert PATH_TO_LLM_IMAGE.endswith(".jpeg") or PATH_TO_LLM_IMAGE.endswith(".png") or PATH_TO_LLM_IMAGE.endswith(".jpg"), "Image must be in JPEG, PNG, or JPG format. or None."

def query_on_text():
    """
    ############################################################################
    Asks GPT-4-Turbo to describe why protobufs are the best thing in the world.
    Specifies custom validation function to ensure the response has the word
        "benefit" in it.
    ############################################################################
    """
    ### SEE: DOCSTRING OF `prompt_LLM` FUNCTION FOR MORE DETAILS
    CHOSEN_MODEL_ID = "AZURE_GPT4_TURBO_MODEL_ID"
    PATH_TO_LLM_IMAGE = None
    DESIRED_FORMAT = None
    PROMPT = "Tell me about protobufs and why they're the best thing in the world."
    MAX_TOKENS = 2048
    RETRY_LIMIT = 2
    VALIDATE_FUNC = lambda x: len(x) >= 300
    
    _meta_check(CHOSEN_MODEL_ID=CHOSEN_MODEL_ID, PATH_TO_LLM_IMAGE=PATH_TO_LLM_IMAGE)
    
    IMAGE_BASE64 = None
    if PATH_TO_LLM_IMAGE is not None:
        IMAGE_BASE64 = encode_image(image_path=PATH_TO_LLM_IMAGE)
    
    client, model_id = configure_openai(model_id_to_use=CHOSEN_MODEL_ID)
    
    _FORCE_RAW = True
    response = prompt_LLM(client=client,
                          model_id=model_id,
                          prompt=PROMPT,
                          base64_image=IMAGE_BASE64,
                          desired_format=DESIRED_FORMAT,
                          max_tokens=MAX_TOKENS,
                          force_raw=_FORCE_RAW,
                          validate_func=VALIDATE_FUNC,
                          num_retry=RETRY_LIMIT)

def query_on_text_from_promptfile():
    """
    ############################################################################
    Same as `query_on_text` function, but with the `PROMPT` variable being read
    from a file.
    ############################################################################
    """
    ### SEE: DOCSTRING OF `prompt_LLM` FUNCTION FOR MORE DETAILS
    CHOSEN_MODEL_ID = "AZURE_GPT4_TURBO_MODEL_ID"
    PATH_TO_LLM_IMAGE = None
    DESIRED_FORMAT = None
    PROMPT = "assets/some_prompt.txt"
    MAX_TOKENS = 2048
    RETRY_LIMIT = 2
    VALIDATE_FUNC = lambda x: len(x) >= 300
    
    _meta_check(CHOSEN_MODEL_ID=CHOSEN_MODEL_ID, PATH_TO_LLM_IMAGE=PATH_TO_LLM_IMAGE)
    
    IMAGE_BASE64 = None
    if PATH_TO_LLM_IMAGE is not None:
        IMAGE_BASE64 = encode_image(image_path=PATH_TO_LLM_IMAGE)
    
    client, model_id = configure_openai(model_id_to_use=CHOSEN_MODEL_ID)
    
    _FORCE_RAW = False  # Since we're reading the raw text from the prompt file.
    response = prompt_LLM(client=client,
                          model_id=model_id,
                          prompt=PROMPT,
                          base64_image=IMAGE_BASE64,
                          desired_format=DESIRED_FORMAT,
                          max_tokens=MAX_TOKENS,
                          force_raw=_FORCE_RAW,
                          validate_func=VALIDATE_FUNC,
                          num_retry=RETRY_LIMIT)

def query_on_image():
    """
    ############################################################################
    Provides an image of a dog to GPT-4o and asks it to describe the image.
    Specifies no custom validation function.
    ############################################################################
    """
    ### SEE: DOCSTRING OF `prompt_LLM` FUNCTION FOR MORE DETAILS
    CHOSEN_MODEL_ID = "AZURE_GPT4o_MODEL_ID"
    PATH_TO_LLM_IMAGE = "assets/dog.jpeg"
    DESIRED_FORMAT = None
    PROMPT = "What is in the image? Make sure your response is at least 300 characters long."
    MAX_TOKENS = 2048
    RETRY_LIMIT = 2
    VALIDATE_FUNC = None
    
    _meta_check(CHOSEN_MODEL_ID=CHOSEN_MODEL_ID, PATH_TO_LLM_IMAGE=PATH_TO_LLM_IMAGE)
    
    IMAGE_BASE64 = None
    if PATH_TO_LLM_IMAGE is not None:
        IMAGE_BASE64 = encode_image(image_path=PATH_TO_LLM_IMAGE)
    
    client, model_id = configure_openai(model_id_to_use=CHOSEN_MODEL_ID)
    
    _FORCE_RAW = True
    response = prompt_LLM(client=client,
                          model_id=model_id,
                          prompt=PROMPT,
                          base64_image=IMAGE_BASE64,
                          desired_format=DESIRED_FORMAT,
                          max_tokens=MAX_TOKENS,
                          force_raw=_FORCE_RAW,
                          validate_func=VALIDATE_FUNC,
                          num_retry=RETRY_LIMIT)

def query_on_image_with_json():
    """
    ############################################################################
    Identical to `query_on_image` function, but with different `PROMPT` and
    `DESIRED_FORMAT` variables. A custom validation function is also specified.
    ############################################################################
    """
    ### SEE: DOCSTRING OF `prompt_LLM` FUNCTION FOR MORE DETAILS
    CHOSEN_MODEL_ID = "AZURE_GPT4o_MODEL_ID"
    PATH_TO_LLM_IMAGE = "assets/dog.jpeg"
    DESIRED_FORMAT = 'json'
    PROMPT = "What is in the image? Ensure your response is in `json` format with two keys: `description` and `confidence`."
    MAX_TOKENS = 2048
    RETRY_LIMIT = 2
    VALIDATE_FUNC = lambda x: all(k in x for k in ("description", "confidence"))
    
    _meta_check(CHOSEN_MODEL_ID=CHOSEN_MODEL_ID, PATH_TO_LLM_IMAGE=PATH_TO_LLM_IMAGE)
    
    IMAGE_BASE64 = None
    if PATH_TO_LLM_IMAGE is not None:
        IMAGE_BASE64 = encode_image(image_path=PATH_TO_LLM_IMAGE)
    
    client, model_id = configure_openai(model_id_to_use=CHOSEN_MODEL_ID)
    
    _FORCE_RAW = True
    response = prompt_LLM(client=client,
                          model_id=model_id,
                          prompt=PROMPT,
                          base64_image=IMAGE_BASE64,
                          desired_format=DESIRED_FORMAT,
                          max_tokens=MAX_TOKENS,
                          force_raw=_FORCE_RAW,
                          validate_func=VALIDATE_FUNC,
                          num_retry=RETRY_LIMIT)

def query_on_image_with_json_and_exceptionvalidation():
    """
    Identical to `query_on_image_with_json` function, but with a custom
    validation function that raises an exception if the response does not
    contain the word "dog". The point of this example is to show that your
    validation function need not return a boolean, but can also raise an
    exception to trigger a retry. If it passes the validation function, it MUST
    return True, though (or any other truthy value).
    """
    ### SEE: DOCSTRING OF `prompt_LLM` FUNCTION FOR MORE DETAILS
    CHOSEN_MODEL_ID = "AZURE_GPT4o_MODEL_ID"
    PATH_TO_LLM_IMAGE = "assets/dog.jpeg"
    DESIRED_FORMAT = 'json'
    PROMPT = "What is in the image? Ensure your response is in `json` format with two keys: `description` and `confidence`."
    MAX_TOKENS = 2048
    RETRY_LIMIT = 2
    def VALIDATE_FUNC(response):
        if not all(k in response for k in ("description", "confidence")):
            raise Exception("Response does not contain the word 'dog'.")
        return True
    
    _meta_check(CHOSEN_MODEL_ID=CHOSEN_MODEL_ID, PATH_TO_LLM_IMAGE=PATH_TO_LLM_IMAGE)
    
    IMAGE_BASE64 = None
    if PATH_TO_LLM_IMAGE is not None:
        IMAGE_BASE64 = encode_image(image_path=PATH_TO_LLM_IMAGE)
    
    client, model_id = configure_openai(model_id_to_use=CHOSEN_MODEL_ID)
    
    _FORCE_RAW = True
    response = prompt_LLM(client=client,
                          model_id=model_id,
                          prompt=PROMPT,
                          base64_image=IMAGE_BASE64,
                          desired_format=DESIRED_FORMAT,
                          max_tokens=MAX_TOKENS,
                          force_raw=_FORCE_RAW,
                          validate_func=VALIDATE_FUNC,
                          num_retry=RETRY_LIMIT)

# You can image a `query_on_text_with_json` function here,
#   similar to `query_on_image_with_json

if __name__ == "__main__":
    print(fmt("Running the testing script...", n="blue"))
    
    print(fmt("Querying GPT-4-Turbo on text literally defined in code...", n="bold"))
    query_on_text()
    print(fmt("Querying GPT-4-Turbo on text defined on the filesystem...", n="bold"))
    query_on_text_from_promptfile()
    print(fmt("Querying GPT-4o on image...", n="bold"))
    query_on_image()
    print(fmt("Querying GPT-4o on image with JSON format request...", n="bold"))
    query_on_image_with_json()
    print(fmt("Querying GPT-4o on image with JSON format request and custom validation that raises an exception...", n="bold"))
    query_on_image_with_json_and_exceptionvalidation()