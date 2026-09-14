"""authpar: The Authorization-Claim Paradox.

A mechanistic study of why self-asserted authorization *increases* refusal on
legitimate dual-use requests. See PLAN.md for the full research design.

The package separates the pipeline into small, testable modules:

- templates : the factorial of framings/channels (H1-H4 disambiguation).
- data      : builds the base-task corpus and the crossed prompt table.
- modeling  : model/tokenizer loading, chat prompts with channel + thinking mode.
- capture   : multi-position residual-stream activation capture.
- refusal   : first-token refusal log-odds metric + greedy generation.
- directions: extraction of d_harm, d_refusal, d_claim, d_persuasion (DiD).
- interventions: ablate / add / patch / weight-orthogonalize operators.
- judge     : response categorization (rule-based, with an LLM-judge seam).
- stats     : bootstrap-by-base-task and clustered effect estimates.

Nothing in the import path requires a GPU or a loaded model, so the data and
analysis stages run on a laptop; only the stages that actually forward the model
need CUDA.
"""

__all__ = [
    "templates",
    "data",
    "modeling",
    "capture",
    "refusal",
    "directions",
    "interventions",
    "judge",
    "stats",
]

__version__ = "0.1.0"
