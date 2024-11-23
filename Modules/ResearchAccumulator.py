import os
from typing import List
import PyPDF2 # camelot instead?
from scholarly import scholarly
from llama_index.llms.openai import OpenAI
from llama_index.readers.papers import ArxivReader

from Schemas.Gaps import Gap
from Schemas.Accumulation import Context

# Need to install 
# pip install scholarly
# pip install --upgrade llama-index-readers-papers

class ResearchAccumulator:
    def __init__(self, storage: str = None, papers_per_search: int = 1):
        # The research that has been accumulated by some external system
        self.research = []
        self.accumulation_agent = None
        self.storage_dir = storage
        self.papers_per_gap = papers_per_search

    def accumulate_from_dir(self, dir):
        # Read the PDFs from the directory
        pdf_files = [f for f in os.listdir(dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dir, pdf_file)
            
            # Extract text from each PDF
            text = self.__extract_text(pdf_path)
            
            # Initialize the Context objects for each PDF
            context = Context(
                paper_id=len(self.research) + 1, # TODO: Is there a better way?
                paper_context=text
                )
            
            # Add the Context objects to the research list
            self.research.append(context)


    def get_paper_titles_from_scholar(self, query: str, num_results=1) -> List[str]:
        search_query = scholarly.search_pubs(query)
        titles = []
        for i, paper in enumerate(search_query):
            if i >= num_results:
                break
            titles.append(paper['bib']['title'])
        return titles


    def accumulate_from_gaps(self, gaps: List[Gap]):
        # Initialize the LLM
        llm = OpenAI(model="gpt-4o-mini")

        # Initialize arxiv reader
        arxiv_reader = ArxivReader()

        new_research = []

        for gap in gaps:
            # Generate a search term using the LLM with a more specific prompt
            prompt = (
                f"As an expert researcher, compose a concise and precise search query "
                f"to find academic papers that address the following research gap:\n\n"
                f"\"{gap.gap_description}\"\n\n"
                f"The search query should include relevant keywords and phrases, and be suitable "
                f"for use in academic databases like Google Scholar or arXiv."
                "Please only separate terms using commas and spaces."
                "Please only include the search query in your response."
            )
            search_term = llm.complete(prompt)
            print(f"Search term: {search_term}")

            # Search for paper titles using Google Scholar
            search_results = self.get_paper_titles_from_scholar(search_term.text, self.papers_per_gap)

            # Get each paper from arxiv
            for title in search_results:

                try:
                    papers = arxiv_reader.load_data(
                        search_query=title,
                        #papers_dir=self.storage_dir,
                        max_results=1
                    )
                except Exception as e:
                    print(f"Error loading paper from arxiv: {e}")
                    continue
                

                for paper in papers:
                    # Add the paper to the research list
                    self.research.append(paper)

                    # Add the paper to the research list
                    context = Context(
                        paper_id=len(self.research) + 1,
                        paper_context=paper.text
                    )

                    new_research.append(context)

        return new_research

    
    def accumulate(self, gaps=None, dir=None, **kwargs) -> List[Context]:
        if dir is not None:
            self.accumulate_from_dir(dir)
        elif gaps is not None:
            return self.accumulate_from_gaps(gaps)
        
        return self.research
    
    def __extract_text(self, pdf_path: str) -> str:
        # Extract the text from the PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    
    ###########################
    ########### CORE ##########
    ###########################
    
    def core(self, gaps: List[Gap] | None) -> List[Context]:
        return self.accumulate(gaps=gaps)
    


if __name__ == '__main__':
    pdf_path = "./resources/validation/pdf"
    ra = ResearchAccumulator()
    ctx = ra.accumulate(dir=pdf_path)
    assert len(ctx) == 3
    assert ctx[0].paper_id == 1
    assert ctx[1].paper_id == 2

    #print(ctx[0].paper_context)

    gaps = [
        Gap(
            gap_id=1,
            gap_name="bayesian safety validation of autonomous vehicles",
            gap_description=("Bayesian inference for safety validation of autonomous "
            "vehicles has the potential to improve safety and reduce the number of accidents.")
        ),

        Gap(
            gap_id=2,
            gap_name="Dimensionality reduction for safety validation of autonomous vehicles",
            gap_description=("Safety validation of autonomous vehicles requires "
            "reasoning about high dimensional data, like images and state trajectories. "
            "Dimensionality reduction techniques can help to simplify this data and make it easier to reason about for safety validation problems.")
        )
    ]

    new_ctx = ra.accumulate(gaps=gaps)