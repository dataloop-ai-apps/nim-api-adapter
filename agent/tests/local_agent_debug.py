import dtlpy as dl
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent"))
from nim_agent import NIMAgent


MODE = "run_agentic"  # "compare", "run", "run_agentic"

agent = NIMAgent(tester_auto_init=False)

if MODE == "compare":
    agent.fetch_models()
    agent.fetch_dataloop_dpks()
    agent.compare()

elif MODE == "run":
    agent.run(
        limit=2,
        open_pr=False,
        skip_docker=True,
    )

elif MODE == "run_agentic":
    agent.run_agentic(
        limit=15,
        open_pr=False,
        skip_docker=True,
        downloadable_preview=True,
    )
