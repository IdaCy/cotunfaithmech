# CoT Unfaithful Mech Interp

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
RationalCircuit/
├── README.md
├── experiments/
│   ├── comparative/          # Experiments using Iván's dataset on comparative questions, GSM8K restoration, and PutnamBench-style tasks.
│   ├── cot/                  # Experiments on forcing chain-of-thought generation and evaluating step-by-step explanations.
│   └── patching/             # Experiments on hidden state patching, circuit/head analysis, and leveraging Eleuther SAEs/transcoders.
├── scripts/
│   ├── run_comparative.py    # Loads Iván’s dataset, queries Gwen-1.5, and checks for both “Is X > Y?” and “Is Y > X?” responses.
│   ├── logit_attribution.py  # Implements logit lens/attribution methods to inspect model decisions before reasoning concludes.
│   ├── generate_cot.py       # Forces chain-of-thought (CoT) generation (step-by-step explanation) from the model.
│   ├── evaluate_cot.py       # Rates CoT outputs for correctness, identifies contradictory steps, and flags hallucinated facts.
│   ├── patch_hidden_states.py# Implements experiments for identifying crucial circuits/heads and patches hidden states from a “faithful” run into an “unfaithful” run.
│   └── analyze_heads.py      # Uses Eleuther SAEs/transcoders (or similar tools) to analyze the features and contributions of identified heads.
├── data/
│   ├── ivan_dataset/
│   ├── gsm8k/
│   └── putnambench/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── 
├── utils/
│   ├── load_model.py
│   ├── 
│   ├── 
│   └── 
├── requirements.txt
└── README.md
```