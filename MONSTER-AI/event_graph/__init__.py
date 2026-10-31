"""Event graph: an explicit, auditable relationship graph (macro drivers, sectors, stocks) stored relationally.

Edges start as expert priors (low confidence, never presented as measured), are re-estimated from returns with
multiple-testing and sign-stability gates, and are traversed with plain dict adjacency (no graph library).
"""
