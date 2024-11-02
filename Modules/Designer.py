from typing import List

from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

from Schemas.Gaps import Hypothesis


class Designer:
    def __init__(self, hypotheses: List[Hypothesis], **kwargs):
        self.hypotheses = hypotheses
        self.designer = OpenAI(model="gpt-4o-mini")
    
    def design_experiments(self, hypotheses: List[Hypothesis], **kwargs):
        prompt_tmpl = PromptTemplate(
            """
            Given the following hypothesis:

            Hypothesis Name: {hypothesis_name}
            Hypothesis Description: {hypothesis_description}

            Design an experiment to test this hypothesis.

            An experiment may include some of the following:
             - Proposed methodologies
             - Data collection strategies
             - Data preprocessing strategies
             - Model architectures
             - Evaluation strategies
             - Potential open-source resources (softwarde, datasets, etc.)
             - Evaluation metrics
             - Baseline models
             - Expected results
             - Potential challenges

             Please be as specific as possible.

            """
        )
        
        def _design(hypothesis: Hypothesis) -> str:
            return self.designer.complete(
                prompt_tmpl.format(
                    hypothesis_name=hypothesis.hypothesis_name,
                    hypothesis_description=hypothesis.hypothesis_description
                )
            )

        return [_design(hypothesis) for hypothesis in hypotheses]
    
    def core(self) -> List[str]:
        return self.design_experiments(self.hypotheses)
    

if __name__ == '__main__':
    from Modules.ResearchAccumulator import ResearchAccumulator
    from Modules.GapFinder import GapFinder

    # Load research from PDFs
    pdf_path = "./resources/validation/pdf"
    ra = ResearchAccumulator()
    ctxs = ra.accumulate(dir=pdf_path)

    # Run the gap finder
    gap_finder = GapFinder(init_contexts=ctxs, k=3, hypothesis_use_index=True)
    hyp = gap_finder.core([])

    # Run the designer
    designer = Designer(hypotheses=hyp)
    designs = designer.core()

    for d in designs:
        print(d)
        print()