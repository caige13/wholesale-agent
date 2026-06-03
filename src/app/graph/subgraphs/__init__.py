"""Subgraphs — self-contained LangGraph agents embedded as nodes in the root graph.

The root order graph is a linear, deterministic pipeline. Anything that needs a
*cycle* (a model-driven tool-calling loop) lives here as its own compiled graph and
is added to the root as a single node, so the parent stays acyclic and readable and
the loop's state (chat ``messages``) stays private to the subgraph.

Today: ``qa_agent`` (the read-only QUESTION-path tool loop) and the ``tools`` it calls.
"""