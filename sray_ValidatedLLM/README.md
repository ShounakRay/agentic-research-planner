# Validated LLM Code

Your typical LLM call but validates the type of the output according to:

1. Specified output type (json or text)
2. Custom Validation function.

> Validation = re-querying the LLM until it gives you an output that conform to the two criteria above.

It's pretty lightweight and doesn't do anything crazy behind the scenes, but feel free to use it in your code as necessary.

## Introduction

Please see body of `if __name__ == "__main__"` in `main.py` to understand different ways `prompt_LLM` can be called.
Supports base64-encoded image ingestion. Function to encode images `encode_image` in `modules/utilities` is also provided.

## High-Level Usage

```bash
# Run this to install the openai package to hit the LLM
python3 -m pip install openai

# [First, ensure credentials exist in CREDENTIALS_PATH defined in `utilities/constants`]
# OR
# [<SET ENVIRONMENT VARIABLES>]
# THEN CALL:
python3 -m main
# OR
python3 -m minimal

# TODO: Need to load in LLM Access keys
#   Just run `python3 -m main` or `python3 -m minimal and you'll get
#   instructions for how to do this. Slack an engineer to key specific API keys
#   if you don't already have them.
```

`main.py` is just the file with different ways of calling `prompt_LLM` the package

`minimal.py` contains a minimal example of calling an LLM, providing it with an image, and querying it for something.
This is just the `query_on_image_with_json` copy-pasted from `main.py`, but condensed down so it doesn't take up much space.
SEE `main.py` for information about what types of inputs you can feed into to `prompt_LLM` and what sort of outputs you can expect.
SEE `minimal.py` and run `python3 -m minimal` to just run the minimal example you see below.

```python
from pprint import pprint
from modules.llm_funcs import prompt_LLM
from modules.utilities import configure_openai, encode_image

if __name__ == "__main__":
    response = prompt_LLM(*configure_openai(model_id_to_use="AZURE_GPT4o_MODEL_ID"),
                          prompt="What is in the image? Ensure your response is in `json` format with two keys: `description` and `confidence`.",
                          base64_image=encode_image(image_path="assets/dog.jpeg"),
                          desired_format='json',
                          max_tokens=2048,
                          force_raw=True,
                          validate_func=lambda x: all(k in x for k in ("description", "confidence")),
                          num_retry=2)
    pprint(response)
```

## More Details on Usage

Here is the docstring of `prompt_LLM` from `modules/llm_funcs` for your reference.
Want more information? Reach out to Shounak with any qeustions.

```python
def prompt_LLM(client: openai.lib.azure.AzureOpenAI,
               model_id: str,
               prompt: str,
               base64_image: Optional[str] = None,
               desired_format: Optional[str] = None,
               max_tokens=DataConstants.MAX_TOKENS_INFERENCE,
               force_raw: bool = False,
               validate_func: object = None,
               num_retry: int = DataConstants.DFT_LLM_RETRY_LIMIT) -> Union[str, dict]:
    """
    
    `prompt_LLM` does basic validation of the response format depending on the
    desired format.

    Args:
        `client` (openai.lib.azure.AzureOpenAI):
            The OpenAI client.

        `model_id` (str):
            The model ID to use.

        `prompt` (str):
            The LLM prompt. Could point to a file or be a raw string.
            > If pointing to a file, `force_raw` should be set to False. If it
              is set to True, then you'd be explicitly passing in the filename
              to the LLM.
            > If pointing to a raw string, `force_raw` should be set to True.
              If it is set to False, then you'd be tell the LLM to load the
              raw-string in as a file, which could break if the raw-string
              doesn't point to a file.
        
        `base64_image` (Optional[str], optional):
            Base64-encoded image string or None if you are not providing image
            to LLM.
            Defaults to None.

        `desired_format` (Optional[str], optional):
            Must be `json` or None (which means a string, not-guaranteed to be
            in any specific format, will be returned).
            Defaults to None.

        `max_tokens` (_type_, optional):
            Maximum tokens in the output.
            Defaults to DataConstants.MAX_TOKENS_INFERENCE.

        `force_raw` (bool, optional):
            SEE `prompt` argument information.
            Defaults to False.

        `validate_func` (object, optional):
            If you want to do more validation, you can pass a function to
            `validate_func`. This could be a lambda function or a regular
            function signature. This function should take the response as an
            argument and return True if the response is valid, False otherwise.
            NOTE: If this validation function ever raises an exception, it
            triggers a retry.
            
            Defaults to None.

        `num_retry` (int, optional):
            Number of tries to retry LLM call if:
            1) Built-in validation fails. This function will fail if the
               response is not in the `desired_format`.
            2) Custom `validate_func` fails, if set. This function will fail if
               the response does not pass the custom validation.
            
            Defaults to DataConstants.DFT_LLM_RETRY_LIMIT.

    Raises:
        `ResponseValidationException`:
            Raised if the response is not in the `desired_format` after all
            specified retries.

    Returns:
        Union[str, dict]:
            Could be a string or a dictionary, depending on `desired_format`.
    """
    # Implemented in code. SEE SOURCE FILE.
    pass
```
