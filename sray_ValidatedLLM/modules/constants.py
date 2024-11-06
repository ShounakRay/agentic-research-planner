class ConstantError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class Constants:
    def __init__(self):
        # Initially, assertions list is empty, and assertions can be added later
        self._assertions = []

    def add_assertion(self, assertion, message: str):
        # Add an assertion to the list
        self._assertions.append({"assertion": assertion, "message": message})
    
    def list_consts(self):
        return {k: self.__dict__[k] for k in self.__dict__.keys() if not k.startswith("_")}

    def validate(self):
        # Validate all assertions in the list
        for assertion in self._assertions:
            if not assertion['assertion']:
                raise ConstantError(f"Constant assertion failed: {assertion['message']}")
            
class DataConstants(Constants):
    MAX_TOKENS_INFERENCE = 4096
    DFT_LLM_RETRY_LIMIT = 2

    DFT_MODEL_IDS_SUPPORTED = ("AZURE_GPT4o_MODEL_ID", "AZURE_GPT4_TURBO_MODEL_ID")
    
    CREDENTIALS_VARS_REQUIRED = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION")
    CREDENTIALS_VARS_ANY_ONE_OF = ("AZURE_GPT4o_MODEL_ID", "AZURE_GPT4_TURBO_MODEL_ID")
    CREDENTIALS_PATH = ".credentials"
    
    def __init__(self):
        super().__init__()
        self.add_assertion(self.MAX_TOKENS_INFERENCE > 0, "MAX_TOKENS_INFERENCE must be greater than 0")
        self.add_assertion(self.DFT_LLM_RETRY_LIMIT > 0, "DFT_LLM_RETRY_LIMIT must be greater than 0")
        self.validate()