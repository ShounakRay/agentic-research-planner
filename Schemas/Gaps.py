from pydantic import BaseModel

from Schemas.Accumulation import Context

class ExperimentalDesign(BaseModel):
    # This is the flowchart we get for each paper
    experimental_design_id: int
    experimental_design_name: str
    experimental_design_description: str

class PaperKnowledge(BaseModel):
    # This is what we feed in to Prompt 2 in Part 2 to get gaps
    paper_id: int
    paper_context: Context
    paper_experimental_design: ExperimentalDesign

class Gap(BaseModel):
    gap_id: int
    gap_name: str
    gap_description: str

class Hypothesis(BaseModel):
    hypothesis_id: int
    hypothesis_name: str
    hypothesis_description: str