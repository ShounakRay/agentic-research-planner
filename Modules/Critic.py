from typing import Callable, Type


class Critic:
    def __init__(self, supported_keys: list[str], **kwargs):
        _payload = {supported_key: "" for supported_key in supported_keys}
        self._supported_keys = supported_keys
        self.punishments = _payload
    
    def overwatch(self, func: Callable, caller: str) -> Type[None]:
        assert caller in self._supported_keys, f"Caller {caller} not supported."
        # Do some critiquing based on the caller_string
        pass
    
    def chastise() -> Type[None]:
        # THis udpates queries
        pass