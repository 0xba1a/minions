"""Train <-> eval loop for repo-learning agents.

Re-exports the public task, outcome, and orchestrator types used to define
an evaluation task and run it in a loop against a training agent.
"""

from .evalTask import EvalOutcome, EvalTask
from .orchestrator import LoopResult, run_train_eval_loop