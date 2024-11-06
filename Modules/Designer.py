from typing import List
from Schemas.Gaps import Hypothesis
from core import critic

class Designer:
    def __init__(self, **kwargs):
        self.hypotheses = None
        self.designer = None
    
    @critic.overwatch
    def design_experiments(self, hypotheses: List[Hypothesis], **kwargs):        
        def _design(hypothesis: Hypothesis) -> str:
            # Design the experiments
            pass
        return [_design(hypothesis) for hypothesis in hypotheses]
    
    def core(self, hypotheses: List[Hypothesis]) -> List[str]:
        self.hypotheses = hypotheses
        return self.design_experiments(self.hypotheses)