"""Figure conventions shared by every evaluation plot.

Colors (truth black, prior grey, posterior teal), nested quantile bands
(5--95 % plus 25--75 %), shared ``Normalize`` objects so prior/posterior panels
never carry different color scales, non-dimensional axis labels (``z/H``,
``u/U_ref``), and solid-cell masking from STL geometry.

The geometry helpers take the STL path and grid as mandatory arguments -- the
repo's data locations are the caller's business, not the library's.

Populated in WP0.2 (move of ``scripts/figspec/style.py`` and the pure geometry
half of ``scripts/figspec/mask.py``).
"""
