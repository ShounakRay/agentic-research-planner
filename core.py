
from pprint import pprint
from typing import List
from Schemas.Accumulation import Context

# FIXME: By having this import line, we are initializing the global agents
# from startup import research_accumulator, gap_finder, designer
import startup

def run(n_loops=5):
    from startup import critic
    
    # This runs the workflow of accumulating research, finding gaps,
    # getting designs, and incorporating critiques `n_loops` times
    
    def _step():
        contexts: List[Context] = startup.research_accumulator.core()
        hypotheses = startup.gap_finder.core(contexts)
        designs = startup.designer.core(hypotheses)
        
        pprint([design for design in designs])
        
        critic.chastise()
        
        n_loops -= 1
    
    # Accumulate the research
    while n_loops > 0:
        _step()
        
    print("Done.")


if __name__ == "__main__":
    # Run from `Modules` directory in some sequence
    run()