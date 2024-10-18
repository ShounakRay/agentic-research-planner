# TODO: Initialize the queries jointly in a class
from Modules.Critic import Critic
from Modules.Designer import Designer
from Modules.GapFinder import GapFinder
from Modules.ResearchAccumulator import ResearchAccumulator


query_object = object()

# Initialize the modules
research_accumulator = ResearchAccumulator()
gap_finder = GapFinder()
designer = Designer()
critic = Critic()