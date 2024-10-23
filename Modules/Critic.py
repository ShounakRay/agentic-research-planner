from functools import wraps
from typing import Callable, Type


class Critic:
    Critique = str
    
    def __init__(self, supported_keys: list[str], **kwargs):
        _payload = {supported_key: "" for supported_key in supported_keys}
        self._supported_keys = supported_keys
        self.punishments = _payload
    
    # def overwatch(self, func: Callable, caller: str) -> Type[None]:
    #     assert caller in self._supported_keys, f"Caller {caller} not supported."
    #     # Do some critiquing based on the caller_string
    #     pass
    
    # A decorator/wrapper function named `overwatch` that takes a function as
    # an argument and returns a function that wraps the input function but also
    # executes some critiquing based on the name of the input function.
    def overwatch(self, func: Callable) -> Type[None]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            caller = func.__name__
            assert caller in self._supported_keys, f"Caller {caller} not supported."
            output = func(*args, **kwargs)
            
            FUNC_NAME, FUNC_DOCSTRING = func.__name__, func.__doc__
            CRITIQUE_GOAL = None #TODO
            OUTPUT_TO_CRITIQUE = None #TODO
            PREVIOUS_CRITIQUE = None #TODO
            
            # TODO: Retrieve the critique
            # TODO: Apply the critique
            
            # TODO: Implement the LLM-call wrapper w/ validation
    
    @staticmethod
    def example_critic_func(previous_critique: Critique,
                            latest_response: str,
                            ) -> Critique:
        # This is an example of a critic function
        pass
    
    def chastise() -> Type[None]:
        # THis udpates queries
        pass