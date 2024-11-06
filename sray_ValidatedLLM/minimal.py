from pprint import pprint
from sray_ValidatedLLM.modules.llm_funcs import prompt_LLM
from sray_ValidatedLLM.modules.utilities import configure_openai, encode_image

if __name__ == "__main__":
    response = prompt_LLM(*configure_openai(model_id_to_use=None, client_host="openai-async"),
                          prompt="Tell me a joke.",
                          base64_image=None,
                          desired_format=None,
                          max_tokens=2048,
                          force_raw=True,
                          validate_func=None,
                          num_retry=2)
    pprint(response)