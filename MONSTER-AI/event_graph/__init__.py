"""Event graph: an explicit, auditable relationship graph (macro drivers, sectors, stocks) stored relationally.

Edges start as expert priors (low confidence, never presented as measured), are re-estimated from returns with
multiple-testing and sign-stability gates, and are traversed with plain dict adjacency (no graph library). A relationship that
stops working is recorded as a break that asserts no direction. Every change of an edge is versioned in an append-only history,
so the graph valid at any past instant can be rebuilt (``GraphRepository.edges_as_of``) for replays.
"""
