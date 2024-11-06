from functools import wraps
from pprint import pprint
from typing import Callable, Type
from sray_ValidatedLLM.modules.llm_funcs import prompt_LLM
from sray_ValidatedLLM.modules.utilities import configure_openai

class Critic:
    Critique = str
    _critiques = {}  # Global dictionary to store critiques

    def __init__(self, prompt_mapping: dict[str, str], **kwargs):
        self._supported_keys = list(prompt_mapping.keys())
        self._prompt_mapping = prompt_mapping
        # Initialize punishments or other payload settings, if needed
        self._critiques = {key: [""] for key in self._supported_keys}
        # DEBUG: Uncomment eventually
        # self.client, self.model_id = configure_openai(model_id_to_use=None, client_host="openai")

    def overwatch(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate the function name
            caller = func.__name__
            assert caller in self._supported_keys, f"Caller '{caller}' not supported."

            # Execute the original function and get the output
            output = func(*args, **kwargs)

            # Prepare information for critiquing
            func_name = func.__name__
            func_docstring = func.__doc__ or "No docstring provided."
            previous_critique = self._get_previous_critique(func_name)
            
            # Generate the prompt for the critique using the docstring and previous critiques
            prompt = (
                f"Function Name: {func_name}\n"
                f"Docstring:\n{func_docstring}\n\n"
                f"Previous Critique: {previous_critique}\n\n"
                f"Provide a critique and suggestions for improvement."
            )

            # FIXME
            # Call the LLM with the prompt to get the critique
            # latest_critique = prompt_LLM(self.client, self.model_id, prompt,
            #                              desired_format=None, max_tokens=2048,
            #                              force_raw=True, validate_func=None, num_retry=2)
            latest_critique = "DEBUGGING"
            self._store_critique(func_name, latest_critique)

            # Output the critique for user feedback, but this could also be logged
            print(f"Critique for {func_name}:\n{latest_critique}\n")

            return output

        return wrapper

    @staticmethod
    def example_critic_func(previous_critique: Critique, latest_response: str) -> Critique:
        # A placeholder example that processes the latest critique
        # This can be customized as needed to filter or validate critiques
        return latest_response if latest_response else "No new critique available."

    def chastise(self) -> None:
        # Update the corresponding prompt for each function with the additional critique
        # TODO:
        pass
    
    def _get_previous_critique(self, func_name: str) -> Critique:
        # Retrieve the last critique if available
        return self._critiques.get(func_name, ["No previous critique"])[-1]

    def _store_critique(self, func_name: str, critique: Critique) -> None:
        # Store the latest critique in the global dictionary
        assert func_name in self._supported_keys, f"Caller '{func_name}' not supported."
        self._critiques[func_name].append(critique)