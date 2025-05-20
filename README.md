# Anonymous Submission for EMNLP 2025

This repository contains the code for simulating survey responses using large language models (LLMs) and analyzing structural bias through IRT and DIF.

## Directory Overview

- 01_data/: Preprocessed data and persona/question prompts  
- 02_simulation_runs/: LLM-specific response generation scripts  
- 03_results/: Collected responses, IRT estimates, and DIF outputs  
- 04_analysis/: Analysis scripts for IRT/DIF and preprocessing  

## Usage

Install dependencies:

    pip install -r requirements.txt

Run simulations and analysis:

    python 02_simulation_runs/LLM_simulation_runs_<model>.py
    python 04_analysis/DIF_LLM_auto.py

(Replace <model> with the desired LLM name.)

## Notes

- Sample data only; full runs may require API access.  
- This repository is anonymized for double-blind review.
