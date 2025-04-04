# CoT Unfaithful Mech Interp

## STATUS - 04/04/25, 12:00

Done:  
- data downloaded  
  - comparative pairs  
  - gsm8k  
  - putnam  
- data new: tips Ivan
- experiment comparative setup  
  - required utility script in `utils`  
  - processing script in `experiment_comparative/run_scripts`
- waiting for first experiment to run  

Next:  
- getting comparative experiment result
- setting up other two data sorts experiemts  

## Plan:

1) take comparative questions from Iván's dataset, run Gwen-1.5, check if it gives yes to both 'Is X > Y?' and 'Is Y > X?' / shows restoration errors in gsm8k / shows shortcuts in putnambench-like problems
(maybe also logit lens or logit attribution for model decisions before reasoning ends?)
2) generate CoT, force to step-by-step explanation, rate each chain for correctness, contradictory steps, hallucinated facts
3) if pattern of unfaithful reasoning, trying guess circuit/set of heads that's important, patching in hidden states from faithful to unfaithful run / use eleuther saes/transcoders to check for features

## Literature:

https://arxiv.org/pdf/2307.13702  
https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-cot  
https://transformer-circuits.pub/2025/attribution-graphs/methods.html  
https://arxiv.org/abs/2503.08679  

https://huggingface.co/EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k  
https://huggingface.co/EleutherAI/skip-transcoder-DeepSeek-R1-Distill-Qwen-1.5B-65k  

## Structure:
```
cotunfaithmech/
├── data/
│   ├── comparative_pairs.jsonl
│   ├── gsm8k.jsonl
│   └── putnambench.jsonl
├── experiment_comparative/
│   ├── output/
│   └── run_scripts/
├── experiment_logit_attribution/
│   ├── output/
│   └── run_scripts/
├── experiment_generate_cot/
│   ├── output/
│   └── run_scripts/
├── experiment_patch_hidden_states/
│   ├── output/
│   └── run_scripts/
├── experiment_analyze_heads/
│   ├── output/
│   └── run_scripts/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── 
├── utils/
│   ├── load_model.py
│   ├── download_gsm8k.py
│   ├── download_ivan_comparative.py
│   └── 
├── requirements.txt
├── environment.yml
└── README.md
```

## Getting the Data

cloning:  
https://github.com/putnambench/putnambench

moving to data/:  
temp_chainscope/chainscope/data/problems/gsm8k.yaml  
temp_chainscope/chainscope/data/math_datasets/filtered_putnambench  
temp_chainscope/chainscope/data/putnam2/minimal_fork_of_putnambench_with_clear_answers.yaml  

