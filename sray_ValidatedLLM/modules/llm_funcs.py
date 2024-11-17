import json
from pprint import pprint
from typing import Any, Optional, Type, Union

import openai
from sray_ValidatedLLM.modules.utilities import fmt, path_exists, path_is_file, wrap_for_unpacking
from sray_ValidatedLLM.modules.constants import DataConstants

import aiohttp
import asyncio
from typing import Union, Optional

class ResponseValidationException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

### HELPER FUNCTIONS

def _safe_response_format(desired_format: Union[str, Type[None]], prompt: str) -> dict:
    RESPONSE_FORMAT = {"response_format": {}}
    if desired_format == "json":
        """
        This following assertion is a result of this excerpt from OpenAI source code:
        ```response_format: An object specifying the format that the model must output. Compatible with
              [GPT-4 Turbo](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) and
              all GPT-3.5 Turbo models newer than `gpt-3.5-turbo-1106`.

              Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the
              message the model generates is valid JSON.

              **Important:** when using JSON mode, you **must** also instruct the model to
              produce JSON yourself via a system or user message. Without this, the model may
              generate an unending stream of whitespace until the generation reaches the token
              limit, resulting in a long-running and seemingly "stuck" request. Also note that
              the message content may be partially cut off if `finish_reason="length"`, which
              indicates the generation exceeded `max_tokens` or the conversation exceeded the
              max context length.
        ```
        """
        assert "json" in prompt, f"Prompt file must contain 'json' for desired format to be 'json'. Your prompt was:\n>>>\n{prompt}\n<<<\n"
        RESPONSE_FORMAT["response_format"] = {"type": "json_object"}
    else:
        RESPONSE_FORMAT = {}
    
    assert isinstance(RESPONSE_FORMAT, dict), "Response format must be a dictionary, not a wrapped tuple like we do once above."
    
    return RESPONSE_FORMAT

def _safe_img_format(base64_image: Optional[str]) -> dict:
    # assert base64_image is not None, "Base64 image is None. Generally supported, but should not ever be the case for this Mickey use-case."
    assert isinstance(base64_image, str) or (base64_image is None), "Base64 image must be a string or None."
    if base64_image is not None:
        assert len(base64_image) > 0, "Base64 image must not be empty."
    
    IMG_CONTENT = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        }
    _desired_arg = None if base64_image is None else IMG_CONTENT
    IMG_CONTENT = wrap_for_unpacking(_desired_arg)
    return IMG_CONTENT

def _safe_prompt_format(prompt: str) -> dict:
    _desired_argument_elem = {"type": "text",
                              "text": prompt}
    return wrap_for_unpacking(_desired_argument_elem)

def _validate_response_format(received_output: Union[str, dict],
                              desired_format: Union[str, Type[None]]) -> Union[str, dict]:
    if not isinstance(received_output, dict) and not isinstance(received_output, str):
        raise ResponseValidationException(f"Received output must be a dictionary or a string. Received type is {type(received_output)}. Surprise, surprise.")
    if desired_format == "json":
        _ERR_MSG = lambda x: f"Desired format is JSON, but received output is not a dictionary. Check model ID and ensure prompt has `json` in it. Output has type {type(x)} with content:\n{x}"
        try:
            received_output = json.loads(received_output)
            if not isinstance(received_output, dict):
                raise Exception("dummy")
        except Exception as e:
            raise ResponseValidationException(_ERR_MSG(received_output))
    return received_output

### CORE FUNCTIONS

def load_prompt(prompt: str,
                substitutions: dict[str, Any] = None,
                force_raw: bool = False) -> str:
    """Load a prompt from a file or a raw string and apply substitutions.

    Args:
        `prompt` (str):
            The path to a file containing the prompt or a raw string as the
            prompt.
        `substitutions` (dict[str, Any], optional):
            A dictionary of substitutions to be applied to the prompt. 
            The keys are phrases to find in the prompt, and the values are the
            substitutions to replace them with.
            Defaults to None.
        `force_raw` (bool, optional):
            If True, treat the prompt as a raw string
            even if it is a file path.
            Defaults to False.

    Returns:
        str: The loaded prompt with substitutions applied, if any.
    """
    
    # Either get prompt from file or from a raw string entered as a raw string.
    if not force_raw:
        assert path_exists(prompt) and path_is_file(prompt), f"Prompt file {prompt} does not exist or is not a file."
        with open(prompt, "r") as f:
            prompt = f.read()
    assert prompt != "", "Prompt from manual user input or from file can't be empty."
    assert isinstance(prompt, str), "Prompt must be a string."
    assert substitutions is None or isinstance(substitutions, dict), "Substitutions must be a dictionary or None."
    assert isinstance(force_raw, bool), "Force raw must be a boolean."
    
    # Apply substitutions to prompt
    if substitutions is not None:
        for phrase_to_find, substitution in substitutions.items():
            assert isinstance(phrase_to_find, str), f"phrase to find in prompt must be a string, got {phrase_to_find}"
            if substitution in ("", {}, None):
                substitution = "[IGNORE: Referred information not available, please ignore this part.]"
            else:
                substitution = json.dumps(substitution, indent=4) if isinstance(substitution, dict) else str(substitution)
            prompt = prompt.replace(phrase_to_find, substitution)
    
    return prompt

# Decorator for showing input and output for `prompt_LLM` function
def show_io(func):
    def wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        _img_str = "and an image" if 'base64_image' in kwargs.keys() else "without any image"
        print(f"For the prompt:\n{fmt(prompt, n='cyan')}\n{_img_str},")
        output = func(*args, **kwargs)
        if output is not None:
            print(f"The response is:")
            fmt("", n="green", just_code=True)
            pprint(output) if isinstance(output, dict) else print(output)
            fmt("", n="TERMINATE", just_code=True)
            print("\n")
        return output
    return wrapper


# async def async_prompt_LLM(**kwargs) -> Union[str, dict]:
#     max_retry = 5
#     output = [""] * len(prompts)
#     while max_retry > 0:
#         try:
#             batch_responses = await asyncio.gather(
#                 *[async_apply(p, **kwargs) for p in prompts]
#             )
#             return batch_responses
#         except Exception as ex:
#             print(ex)
#             return output

def prompt_LLM(client: Union[openai.lib.azure.AzureOpenAI, openai.OpenAI],
               model_id: str,
               prompt: str,
               base64_image: Optional[str] = None,
               desired_format: Optional[str] = None,
               max_tokens=DataConstants.MAX_TOKENS_INFERENCE,
               force_raw: bool = False,
               validate_func: object = None,
               num_retry: int = DataConstants.DFT_LLM_RETRY_LIMIT,
               system_prompt: str = None) -> Union[str, dict]:
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
    
    # TODO: Add an assertion cross-referencing the model_id with desired_format.
    #       JSON is supposedly only available for certain GPT models, some of
    #       which we may not host on Azure, need to double check.
    
    # assert isinstance(client, openai.lib.azure.AzureOpenAI), "Client must be an instance of the OpenAI client."
    assert isinstance(prompt, str), "Prompt must be a string, representing a file path or a raw prompt string."
    assert isinstance(base64_image, str) or base64_image is None, "Base64 image must be a string or None."
    if base64_image is not None:
        assert len(base64_image) > 0, "Base64 image must not be empty."
    assert isinstance(max_tokens, int) and (0 < max_tokens <= DataConstants.MAX_TOKENS_INFERENCE), f"Max tokens must be an integer between 0 and {DataConstants.MAX_TOKENS_INFERENCE}."
    assert desired_format == "json" or desired_format is None, "Desired format must be 'json' or None. Allows for experimentation."
    assert isinstance(force_raw, bool), "Force raw must be a boolean."
    assert validate_func is None or callable(validate_func), "Validate function must be a callable function or None."
    if not (isinstance(num_retry, int) and num_retry > 0):
        print(fmt("No retries are left to execute LLM call.", n="red"))
        return None
    
    prompt = load_prompt(prompt, force_raw=force_raw)
    
    IMG_CONTENT = _safe_img_format(base64_image)
    PROMPT_CONTENT = _safe_prompt_format(prompt)
    RESPONSE_FORMAT = _safe_response_format(desired_format, prompt)
    
    _SYSTEM_PROMPT = {
        "role": "system",
        "content": system_prompt
    } if system_prompt else None
    _SYSTEM_PROMPT = wrap_for_unpacking(_SYSTEM_PROMPT)
    
    # TODO: Minor, make these LLM args parameters
    async def asynch_response(max_tokens, model_id):
        return await client.chat.completions.create(
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [*PROMPT_CONTENT, *IMG_CONTENT],
                },
                *_SYSTEM_PROMPT
            ],
            **RESPONSE_FORMAT
        )
    
    def synch_response(max_tokens, model_id):
        return client.chat.completions.create(
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [*PROMPT_CONTENT, *IMG_CONTENT],
                },
                *_SYSTEM_PROMPT
            ],
            **RESPONSE_FORMAT
        )
    
    if isinstance(client, openai.AsyncOpenAI):
        print(fmt("Asynchronous call to LLM.", n="green"))
    else:
        print(fmt("Synchronous call to LLM.", n="green"))
    response = asyncio.run(asynch_response(max_tokens, model_id)) if isinstance(client, openai.AsyncOpenAI) else synch_response(max_tokens, model_id)
    
    print(fmt("Response received from LLM.", n="green"))
    try:
        response = _validate_response_format(response.choices[0].message.content,
                                             desired_format=desired_format)
        custom_output: bool = validate_func(response) if validate_func is not None else True
        if not custom_output:
            raise ResponseValidationException(f"Custom validation failed for response:\n{response}")
    except (Exception) as rve:
        print(fmt(f"Validation failed. RETRY {DataConstants.DFT_LLM_RETRY_LIMIT - num_retry + 1} of {num_retry}.", n="yellow"))
        print(rve)
        # Retry the function call, if any retries are left
        if num_retry > 0:
            num_retry = num_retry - 1
            return prompt_LLM(client=client,
                              model_id=model_id,
                              base64_image=base64_image,
                              desired_format=desired_format,
                              max_tokens=max_tokens,
                              prompt=prompt,
                              force_raw=force_raw,
                              validate_func=validate_func,
                              num_retry=num_retry)
        else:
            msg = fmt(f"Validation failed after {DataConstants.DFT_LLM_RETRY_LIMIT} retries. Last error: {rve}", n="red")
            raise ResponseValidationException(msg)
    return response