# flow-alignment-thesis

Python code developed as part of a master’s thesis on alignment of multi-tube flow cytometry data.

The repository contains the implementation of a per-marker alignment strategy based on
density-derived landmarks and monotone piecewise-linear transformations, with a quantile-based
fallback for selected markers.

## Files
- `src/norm_functions.py`: landmark extraction and alignment functions  
- `run_alignment.py`: script used to run the alignment pipeline
