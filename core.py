
from pprint import pprint
from typing import List
from Schemas.Accumulation import Context

from Modules.Critic import Critic

# Initialize the critic only
prompt_mapping = {
    "_get_top_k_papers": "Prompts/raccum_prompt.txt",
    "find_gaps": "Prompts/gapfinder_prompt1.txt",
    "get_hypotheses": "Prompts/gapfinder_prompt2.txt",
    "design_experiments": "Prompts/designer_prompt1.txt"
}
critic = Critic(prompt_mapping=prompt_mapping)


# FIXME: By having this import line, we are initializing the global agents
# from startup import research_accumulator, gap_finder, designer
# import startup

def run(n_loops=5):
    from Modules.Designer import Designer
    from Modules.GapFinder import GapFinder
    from Modules.ResearchAccumulator import ResearchAccumulator
    
    # This runs the workflow of accumulating research, finding gaps,
    # getting designs, and incorporating critiques `n_loops` times
    
    research_accumulator = ResearchAccumulator()
    gap_finder = GapFinder(k=5)
    designer = Designer()
    
    def _step():
        contexts: List[Context] = research_accumulator.core()
        hypotheses = gap_finder.core(contexts)
        designs = designer.core(hypotheses)
        
        pprint([design for design in designs])
        
        critic.chastise()
        
    # Accumulate the research
    while n_loops > 0:
        print(f"> Loop {n_loops}...")
        _step()
        n_loops -= 1
        
    print("Done.")
    
    pprint(critic._critiques)


if __name__ == "__main__":
    # Run from `Modules` directory in some sequence
    run()