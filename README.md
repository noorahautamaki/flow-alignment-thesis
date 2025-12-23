# flow-alignment-thesis
Python implementation of a flow cytometry alignment method developed for a master’s thesis.

## Overview
This repository implements a per-marker alignment strategy based on density-derived landmarks
and monotone piecewise-linear transformations, with quantile-based fallback for selected markers.

## Structure
- `src/norm_functions.py`: core landmark extraction and alignment utilities
- `run_alignment.py`: main script used to run the alignment pipeline

## Usage
```bash
python run_alignment.py --raw-dir <path> --ref-path <path> --out-dir <path>
