from typing import List

from Schemas.Accumulation import Context


class ResearchAccumulator:
    def __init__(self):
        # The research that has been accumulated by some external system
        self.research = None
        self.accumulation_agent = None
    
    def accumulate(self, **kwargs) -> List[Context]:
        # Accumulate the research
        pass
    
    def __extract_text(self, pdf_path: str) -> str:
        # Extract the text from the PDF
        pass
    
    ###########################
    ########### CORE ##########
    ###########################
    
    def core(self) -> List[Context]:
        return self.accumulate()