from typing import List
from Schemas.Gaps import Hypothesis


class Designer:
    def __init__(self, hypotheses: List[Hypothesis], **kwargs):
        self.hypotheses = hypotheses
        self.designer = None
    
    def design_experiments(self, hypotheses: List[Hypothesis], **kwargs):
        def _design(hypothesis: Hypothesis) -> str:
            # Design the experiments
            pass
        return [_design(hypothesis) for hypothesis in hypotheses]
    
    def core(self) -> List[str]:
        return self.design_experiments(self.hypotheses)