import os
from typing import List
import PyPDF2 # camelot instead?

from Schemas.Accumulation import Context

class ResearchAccumulator:
    def __init__(self):
        # The research that has been accumulated by some external system
        self.research = []
        self.accumulation_agent = None

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
    
    def accumulate(self, dir=None, **kwargs) -> List[Context]:
        if dir is not None:
            self.accumulate_from_dir(dir)
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
    
    def core(self) -> List[Context]:
        return self.accumulate()
    


if __name__ == '__main__':
    pdf_path = "./resources/validation/pdf"
    ra = ResearchAccumulator()
    ctx = ra.accumulate(dir=pdf_path)
    assert len(ctx) == 3
    assert ctx[0].paper_id == 1
    assert ctx[1].paper_id == 2

    print(ctx[0].paper_context)