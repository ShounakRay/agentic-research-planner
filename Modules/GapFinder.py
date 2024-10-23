from typing import List, Type
from Modules import Critic
from Schemas.Accumulation import Context
from Schemas.Gaps import ExperimentalDesign, Hypothesis, Gap
from startup import critic

# TODO: What's our vector store interface?

class GapFinder:
    def __init__(self, k: int, **kwargs):
        # The gaps that have been found by some external system
        self.gaps = None
        self.gap_finding_agent = None
        self.vector_store = self.__init_vector_store()
        self.k = None
    
    ###########################
    ####### HOUSEKEEPING ######
    ###########################
    
    def __init_vector_store(self, **kwargs) -> Type[None]:
        """Initializes the vector store for the GapFinder based on procured
            research.

        Returns:
            Type[None]: _description_
        """
        # TODO: Initialize the vector store / ChromaDB stuff
        pass
    
    ###########################
    ########## CORE ###########
    ###########################
    
    @critic.overwatch
    def find_gaps(self, contexts: List[Context], flows: List[ExperimentalDesign], **kwargs) -> List[Gap]:
        """Given context from the accumulated research, this gets the gaps.

        Args:
            contexts (List[Context]): This is the output of `_get_top_k_papers`.
            flows (List[ExperimentalDesign]): This is the output of `convert_papers_to_flowcharts`.

        Returns:
            List[Gap]: _description_
        """
        def _find(context: Type[Context], flow: Type[ExperimentalDesign]) -> Gap:
            # Find the gaps
            pass
        return [_find(context, flow) for context, flow in zip(contexts, flows)]
    
    @critic.overwatch
    def get_hypotheses(self, Gaps: List[Gap]) -> List[Hypothesis]:
        """Given the gaps, it returns the hypotheses (a simple transformation).

        Args:
            Gaps (List[Gap]): _description_

        Returns:
            List[Hypothesis]: _description_
        """
        def _get(gaps: Type[Gaps]) -> Hypothesis: # type: ignore
            # Get the hypotheses
            pass
        
        return [_get(gap) for gap in Gaps]
    
    def convert_papers_to_flowcharts(self, contexts: List[Context], **kwargs) -> List[ExperimentalDesign]:
        """Given the contexts, it converts the papers to flowcharts.

        Args:
            contexts (List[Context]): _description_
        """
        def _convert(context: Type[Context]) -> ExperimentalDesign:
            # Convert the papers to flowcharts
            pass
        
        return [_convert(context) for context in contexts]
    
    ###########################
    ######### HELPERS #########
    ###########################
    
    # FIXME: Make sure this function header actually works with this usage
    @critic.overwatch
    def _get_top_k_papers(self, query: str, top_k: int, **kwargs) -> List[Context]:
        """This gets the top k papers from the accumulated research. Searches
        over the vector store for the most relevant papers.

        Returns:
            List[Context]: _description_
        """
        
        # TODO: Make this it's own file later
        query = """
        Given the accumulated research and the following hypothesis, find the
        most relevant papers.
        
        {{research}}
        {{hypothesis}}
        """
        
        # Get the top k papers
        return [""]
    
    def _adds_papers_to_store(self, papers: List[Context], **kwargs) -> Type[None]:
        """Adds the papers to the vector store.

        Args:
            papers (List[Context]): _description_
        """
        # Add the papers to the vector store
        pass
    
    ###########################
    ########### CORE ##########
    ###########################
    
    def core(self, paper_contexts: List[Context]) -> List[Hypothesis]:
        self._adds_papers_to_store(paper_contexts)
        top_k_papers: List[Context] = self._get_top_k_papers(query="Some query", top_k=self.k)
        get_flowcharts: List[ExperimentalDesign] = self.convert_papers_to_flowcharts(top_k_papers)
        gaps: List[Gap] = self.find_gaps(top_k_papers, get_flowcharts)
        hypotheses: List[Hypothesis] = self.get_hypotheses(gaps)
        
        return hypotheses