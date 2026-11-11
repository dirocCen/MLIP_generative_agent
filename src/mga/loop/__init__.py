"""Closed search loop: L0 descriptors -> surrogate -> L1 batch -> retrain.

``data``       builds one table per pool (L0 features, L1 labels, costs).
``surrogate``  Gaussian-process surrogate of log10 kappa_L1 and the acquisition strategies.
``replay``     replays strategies on a fully labelled subset and records the curves.
"""
