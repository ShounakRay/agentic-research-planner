import re
from typing import List, Type

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

# from Modules import Critic
from Schemas.Accumulation import Context
from Schemas.Gaps import ExperimentalDesign, Hypothesis, Gap
from startup import critic

# TODO: What's our vector store interface?

def response_to_gaps(input_string: str) -> List[Gap]:
    """Converts the response from the LLM to a list of gaps.

    Args:
        response (str): The response from the LLM.

    Returns:
        List[Gap]: The gaps that were found.
    """
    responses = input_string.strip().split('---------------------')
    
    gaps = []
    for response in responses:
        # Extract the values using regular expressions
        gap_id = int(re.search(r'gap_id=(\d+)', response).group(1))
        gap_name = re.search(r"gap_name='([^']*)'", response).group(1)
        gap_description = re.search(r"gap_description='([^']*)'", response).group(1)
        
        # Create a Gap object and add it to the list
        gap = Gap(gap_id=gap_id, gap_name=gap_name, gap_description=gap_description)
        gaps.append(gap)
    
    return gaps

class GapFinder:
    def __init__(
            self, k: int,
            init_contexts: List[Context],
            chunk_size=100,
            hypothesis_use_index=False,
            **kwargs
            ):
        # The gaps that have been found by some external system
        self.gaps = None
        self.gap_finding_agent = OpenAI(model="gpt-4o-mini")
        self.chunk_size = chunk_size
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.__init_vector_store(init_contexts)
        self.k = k  # Correctly assign the value of k
        self.hypothesis_use_index = hypothesis_use_index
    
    ###########################
    ####### HOUSEKEEPING ######
    ###########################
    
    def __init_vector_store(self, init_contexts : List[Context], **kwargs) -> Type[None]:
        """Initializes the vector store for the GapFinder based on procured
            research.

        Returns:
            Type[None]: _description_
        """
        # TODO: Initialize the vector store / ChromaDB stuff
        self.vector_store = ChromaVectorStore(
            chroma_collection=chromadb.Client().create_collection("research")
        )
        #self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        init_docs = [Document(text=ctx.paper_context, metadata={"paper_id": ctx.paper_id}) for ctx in init_contexts]
        
        self.pipeline = IngestionPipeline(
            transformations=[
                #SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=0),
                TokenTextSplitter(chunk_size=1024, chunk_overlap=128),
                self.embed_model,
            ],
            vector_store=self.vector_store,
        )

        self.pipeline.run(documents=init_docs)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=self.embed_model)
    
    ###########################
    ########## CORE ###########
    ###########################
    
    @critic.overwatch
    def find_gaps(self, **kwargs) -> List[Gap]:
        """Given context from the accumulated research, this gets the gaps.

        Args:
            contexts (List[Context]): This is the output of `_get_top_k_papers`.
            flows (List[ExperimentalDesign]): This is the output of `convert_papers_to_flowcharts`.

        Returns:
            List[Gap]: _description_
        """
        # def _find(context: Type[Context], flow: Type[ExperimentalDesign]) -> Gap:
        #     # Find the gaps
        #     pass

        # return [_find(context, flow) for context, flow in zip(contexts, flows)]
        
        gap_finder_prompt = """
        Please identify potential research gaps, opportunities, or areas for further investigation based on the given papers. Please include citations for each claimed gap. Be as specific as possible.
        """
        query_engine = self.index.as_query_engine(
            output_cls=Gap,
            response_mode="accumulate",
            llm=self.gap_finding_agent
        )
        response = query_engine.query(gap_finder_prompt).response

        return response_to_gaps(response)
    
    @critic.overwatch
    def get_hypotheses(self, gaps: List[Gap]) -> List[Hypothesis]:
        """Given the gaps, it returns the hypotheses (a simple transformation).

        Args:
            Gaps (List[Gap]): _description_

        Returns:
            List[Hypothesis]: _description_
        """
        prompt_tmpl = PromptTemplate(
            """
            Can you please help me generate a research hypothesis based on the following research gap that we've identified?

            A hypothesis may include the following:
             - A proposed explanation for a phenomenon
             - A proposed novel approach for solving a problem
             - A novel algorithm, method, or combination of methods
             - A novel problem formulation

            Please be as specific as possible. Here is the research gap:

            {gap_description}
            """
        )

        def _get(gap: Type[Gap]) -> Hypothesis: # type: ignore
            hyp = self.gap_finding_agent.as_structured_llm(Hypothesis).complete(
                prompt_tmpl.format(gap_description=gap.gap_description)
            ).raw
            return hyp
        
        def _get_with_index(gap: Type[Gap]) -> Hypothesis: # type: ignore
            llm = self.gap_finding_agent.as_structured_llm(Hypothesis)
            query_engine = self.index.as_query_engine(llm=llm)
            r = query_engine.query(
                prompt_tmpl.format(gap_description=gap.gap_description)
            )
            return r.response
        
        hyp_fn = _get_with_index if self.hypothesis_use_index else _get
        
        return [hyp_fn(gap) for gap in gaps]
    
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
        # if papers is empty, do nothing
        if not papers:
            return
        
        # Add the papers to the vector store
        # each context contains a paper_id and paper_context
        docs = [Document(text=ctx.paper_context, metadata={"paper_id": ctx.paper_id}) for ctx in papers]
        self.pipeline.run(documents=docs)

    
    ###########################
    ########### CORE ##########
    ###########################
    
    def core(self, paper_contexts: List[Context]) -> List[Hypothesis]:
        self._adds_papers_to_store(paper_contexts)
        # top_k_papers: List[Context] = self._get_top_k_papers(query="Some query", top_k=self.k)
        # flowcharts: List[ExperimentalDesign] = self.convert_papers_to_flowcharts(top_k_papers)
        
        # Initially, each of these is a single LLM query
        gaps: List[Gap] = self.find_gaps() # Use vector store
        hypotheses: List[Hypothesis] = self.get_hypotheses(gaps) # Maybe use vector store
        
        return hypotheses
    

if __name__ == '__main__':
    from Modules.ResearchAccumulator import ResearchAccumulator

    # Load research from PDFs
    pdf_path = "./resources/validation/pdf"
    ra = ResearchAccumulator()
    ctxs = ra.accumulate(dir=pdf_path)

    # Run the gap finder
    gap_finder = GapFinder(init_contexts=ctxs, k=3, hypothesis_use_index=True)
    hyp = gap_finder.core([])

    for h in hyp:
        print(h.hypothesis_description)
