# Rhetorical Strategy Extraction and Modeling in U.S. Presidential Debates

This repository contains the code and data for the master's thesis *"Development of a method for automatic extraction and modeling of rhetorical strategy interactions in U.S. presidential debates (1960–2024)"*.

The project has two parts:
1. **Automatic annotation** of debate utterances with rhetorical strategies using an LLM ensemble
2. **Sequential modeling** of rhetorical strategy interactions using Markov chains and graph analysis

---

## Repository structure

```
├── data/
│   ├── unlabeled_df.csv             # Corpus before annotation (utterances + metadata)
│   ├── gold_standard.csv            # Manually annotated gold standard (604 utterances, 4 debates)
│   └── final_corpus.csv             # Fully annotated corpus (42,284 utterances)
│
├── annotation/
│   ├── zero_shot.ipynb              # Zero-shot annotation with 4 LLMs + evaluation on gold standard
│   ├── few_shot.ipynb               # Few-shot annotation with 4 LLMs + evaluation on gold standard
│   ├── cohere_annotate.py           # Cohere few-shot annotation script
│   ├── mistral_fs_annotation.ipynb  # Mistral few-shot annotation
│   ├── model_agreement.ipynb        # Inter-model agreement analysis
│   ├── ensemble.ipynb               # Ensemble evaluation across all 24 model pair configurations
│   └── tie-break_annotation.ipynb   # Final annotation pipeline: Mistral-fs + Cohere-fs + Anthropic tie-break
│
└── modeling/
    └── modeling.ipynb               # Markov chain and graph analysis of rhetorical strategy transitions
```

---

## Data

### `unlabeled_df.csv`
The corpus before annotation. Each row is one utterance. Columns: `title`, `link`, `election_type`, `date`, `speaker`, `place`, `text`, `order`.

Covers 174 debates (125 primary, 49 general) from 1960 to 2024, collected from [The American Presidency Project](https://www.presidency.ucsb.edu).

### `gold_standard.csv`
Manually annotated gold standard used for evaluating automatic annotation. Contains 604 utterances from 4 debates:
- Vice Presidential Debate in Houston, TX (1976)
- Democratic Candidates Debate in Manchester, NH (2000, primary)
- Presidential Debate at Hofstra University, NY (2008)
- Vice Presidential Debate in New York City (2024)

### `final_corpus.csv`
The fully annotated corpus produced by the final ensemble. Columns include all metadata fields plus: `cohere_strategy1/2`, `mistral_strategy1/2`, `ensemble_strategy1/2`, `ensemble_source` (consensus / anthropic→cohere / anthropic→mistral).

---

## Annotation pipeline

The annotation treats rhetorical strategy labeling as a classification task. Each utterance is sent to an LLM with a structured prompt containing the taxonomy, decision rules, and (in few-shot mode) 10  examples. The model returns a JSON object with `strategy1`, `strategy2`, `confidence`, and `explanation`.

**Taxonomy** (4 content strategies + 2 auxiliary labels):
- `presentation` — positive self-presentation
- `accusation` — negative other-presentation
- `self-justification` — defense or rejection of responsibility
- `appeal` — call to audience or invocation of shared values
- `no_strategy` — too short, interrupted, or lacking context
- `-` — moderator utterances

The full prompts are included in `zero_shot.ipynb` and `few_shot.ipynb`.

**Models compared**: Mistral Small, Cohere Command-A, OpenAI GPT-5, Anthropic Claude Sonnet 4.5 — in zero-shot and few-shot regimes (8 configurations total).

**Final ensemble**: Mistral few-shot + Cohere few-shot, with Anthropic as tie-break model. Achieves strict strategy1 accuracy of 0.856 on the gold standard.

### Notebook order
1. `zero_shot.ipynb` — run zero-shot annotation, evaluate on gold standard
2. `few_shot.ipynb` — run few-shot annotation, evaluate on gold standard
3. `model_agreement.ipynb` — analyze inter-model agreement
4. `ensemble.ipynb` — evaluate all 24 ensemble configurations on gold standard
5. `mistral_fs_annotation.ipynb` + `cohere_annotate.py` — annotate full corpus
6. `tie-break_annotation.ipynb` — apply tie-break, produce `final_corpus.csv`

---

## Modeling

`modeling.ipynb` builds transition matrices and directed graphs from the annotated corpus and computes:
- Same-speaker and cross-speaker transition matrices
- Matrices for corpus slices: general vs primary, Republican vs Democratic primaries, 1960-2004 vs 2008-2024
- Structural graph metrics: weighted in-degree, self-loop strength, attraction ratio, edge asymmetry, reciprocity
- Stationary distribution π and row entropy
- Likelihood-ratio test for the first-order Markov assumption
- Chi-square test of homogeneity for comparing matrices across groups

---

## Requirements

```
pandas numpy matplotlib seaborn scikit-learn scipy networkx plotly
openai anthropic cohere tqdm
```

API keys for OpenAI, Anthropic, Cohere, and Mistral are required for annotation notebooks. Set them as environment variables or enter via `getpass` when prompted.
