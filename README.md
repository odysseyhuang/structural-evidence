Official implementation of **“Deliberately Incomplete Structural Facts: Accuracy-Efficiency Trade-offs in Repository-Level Code Completion”**.

This repository implements a unified context modeling approach for repository-level code completion. The method retrieves repository evidence from multiple query views, organizes heterogeneous code relations in a unified context graph, and uses multi-path evidence to select compact cross-file context for code generation.

> This repository contains the implementation of **Work I only**.  
> The evidence-quality dataset and Controller introduced in Work II are not included here.

## Overview

Repository-level code completion requires information distributed across multiple files. Pure lexical retrieval may return code that is textually similar but semantically incomplete, while a single dependency relation cannot cover all forms of useful repository context.

Our method contains three main stages:

1. **Multi-View Querying and Candidate Retrieval**  
   Construct complementary queries from the visible left context.

2. **Structural Evidence Representation and Candidate Expansion**  
   Retrieve candidate code blocks using enhanced BM25 and rank fusion.

3. **Structure-Relation-Driven Candidate Matching.**  
   Connect repository evidence through same-file, identifier, import, API-call, receiver-type, member-access, inheritance, override, signature, and visible control/data-flow relations.

## Main Features

- Multi-view retrieval from code context, identifiers, imports, and API calls.
- Unified modeling of lexical, structural, and semantic repository relations.
- Multi-evidence aggregation for candidate code blocks.
- Evidence gating and path-aware reranking.
- Support for Python and Java.
- Evaluation on CrossCodeEval and RepoEval.
- Reproduction scripts for the main experiment, ablations, cross-model experiments, context-window experiments, and significance analysis.

## Repository Structure

```text
.
├── main.py                         # Main training and evaluation entry point
├── generator.py                    # Code-generation model wrapper
├── retriever.py                    # UniXcoder/RLRetriever wrapper
├── datasets.py                     # Dataset loading and example construction
├── bm25.py                         # Candidate segmentation and BM25 retrieval
├── context_query.py                # Multi-view query construction
├── context_candidates.py           # Multi-path recall and rank fusion
├── context_gate.py                 # Candidate and retrieved-context filtering
├── context_graph.py                # Graph expansion, scoring, and tracing
├── unified_context_graph.py        # Unified semantic-relation extraction
├── typed_dependency_graph.py       # Typed dependency extraction
├── utils/                          # Evaluation and model utilities
└── results/                        # Paper-level result summaries
```

## Environment

The experiments require Python, PyTorch, Transformers, and one CUDA-capable GPU.

```bash
conda create -n ucm python=3.10 -y
conda activate ucm
pip install -r requirements.txt
```

The versions used by the released environment are recorded in `requirements.txt`.

The precompiled Tree-sitter parsers under `utils/build/` target Linux x86-64. They must be rebuilt when using another operating system or architecture.

## Data

Download the evaluation data from [Data4RLCoder](https://huggingface.co/datasets/nov3630/Data4RLCoder) and arrange it as follows:

```text
data/
├── cceval/
│   ├── python/
│   │   ├── test.parquet
│   │   └── test.jsonl
│   └── java/
│       ├── test.parquet
│       └── test.jsonl
├── repoeval/
│   ├── line_level/
│   │   ├── test_0.parquet
│   │   ├── test_1.parquet
│   │   └── test.jsonl
│   └── api_level/
│       ├── test_0.parquet
│       ├── test_1.parquet
│       └── test.jsonl
└── github_repos/                   # Only required for GitHubEval
    ├── python/train.parquet
    └── java/train.parquet
```

The datasets are not redistributed in this repository. Please follow the licenses and usage conditions of the original datasets.

## Models

Place the generator and retriever checkpoints under `models/`:

```text
models/
├── deepseek-coder-6.7b-base/
├── CodeLlama-7b-hf/
├── starcoderbase-7b/
├── unixcoder-base/
└── RLRetriever/
```

The final 4K experiment uses:

- Generator: DeepSeek-Coder-6.7B-base
- Retriever: RLRetriever
- Prompt budget: 4,096 tokens
- Cross-file context: 3,072 tokens
- In-file context: 1,024 tokens
- Maximum generation length: 64 tokens
- Number of final retrieved blocks: 10

Model checkpoints are not committed to GitHub. Please download them from their original model pages or the checkpoint link accompanying the paper.

## Reproducing the Main Experiment

The final configuration is stored in:

```text
scripts/submit_ucm_graph_g5_full_4k.sh
```

First inspect the generated Slurm command:

```bash
bash scripts/submit_ucm_graph_g5_full_4k.sh --dry-run
```

To evaluate the four paper benchmarks:

```bash
export PROJECT_DIR="$(pwd)"
export CONDA_ENV="/absolute/path/to/ucm/environment"
export EVAL_DATASETS="cceval_python:cceval_java:repoeval_line:repoeval_api"

bash scripts/submit_ucm_graph_g5_full_4k.sh
```

The default Slurm settings request one GPU. Adjust the partition, GPU type, CPU count, memory, and time limit in `scripts/submit_ucm_a800.slurm` for your cluster.

To evaluate only one dataset:

```bash
EVAL_DATASETS="cceval_python" \
PROJECT_DIR="$(pwd)" \
CONDA_ENV="/absolute/path/to/ucm/environment" \
bash scripts/submit_ucm_graph_g5_full_4k.sh
```

## Ablation Studies

Inspect all ablation jobs:

```bash
bash scripts/submit_ucm_graph_g5_ablation_4k.sh --dry-run
```

Submit the ablations:

```bash
PROJECT_DIR="$(pwd)" \
CONDA_ENV="/absolute/path/to/ucm/environment" \
EVAL_DATASETS="cceval_python:cceval_java:repoeval_line:repoeval_api" \
bash scripts/submit_ucm_graph_g5_ablation_4k.sh
```

The script evaluates the contribution of multi-view retrieval, unified graph relations, multi-evidence aggregation, path reranking, and evidence gating.

## Cross-Model Evaluation

Run the final configuration with CodeLlama-7B and StarCoderBase-7B:

```bash
bash scripts/submit_ucm_graph_g5_cross_model_4k.sh --dry-run
bash scripts/submit_ucm_graph_g5_cross_model_4k.sh
```

Model paths can be changed through environment variables or directly in the submission script.

## Context-Window Evaluation

Inspect the 8K and 16K context-window experiments:

```bash
bash scripts/submit_ucm_graph_g5_context_matrix.sh --dry-run
```

Submit them with:

```bash
bash scripts/submit_ucm_graph_g5_context_matrix.sh
```

StarCoderBase is restricted to its native context limit. The corresponding script reserves space for the generated tokens.


## Statistical Significance

We use paired bootstrap resampling over per-example exact-match outcomes:

```bash
python scripts/paired_em_bootstrap.py \
    --ours-root result_infer/G5_F1_unified_context_graph_full_all_4k \
    --rlcoder-root result_infer/RLCoder_deepseekcoder_7b_crossfile_3072_infile_1024 \
    --aligncoder-root result_infer/AlignCoder_deepseekcoder_7b_crossfile_3072_infile_1024 \
    --rounds 10000 \
    --seed 123 \
    --output-dir result_analysis/statistical_significance
```

The released summaries are available in:

```text
results/statistical_significance/
```

Per-example predictions required to recompute the analysis are distributed through the project release page when their size or benchmark license makes direct GitHub storage unsuitable.

## Output Files

Each evaluation dataset produces:

```text
<output_dir>/<dataset>/
├── prediction.jsonl
├── prediction_truncated.jsonl
├── detailed_results.json
├── exact_match_idx.jsonl
├── results.json
└── ucm_retrieval_trace.jsonl
```

`ucm_retrieval_trace.jsonl` records the retrieval sources, graph expansions, supporting relations, path scores, and final selected evidence. It is intended for debugging and case-study analysis.

## Reproducibility Notes

- Retrieval queries use only information visible before the completion location.
- Ground-truth code is used only for evaluation.
- Random seeds are fixed in the released implementation.
- The paper’s primary result uses the final G5 configuration without combining metrics from other runs.
- Runtime numbers should be compared only when hardware, model, batch size, prompt budget, dataset, and timing boundaries are identical.
- Generated predictions and `run_config.json` should be retained for every reported experiment.

## Acknowledgements

This implementation is built on the public code and data infrastructure of [RLCoder](https://github.com/DeepSoftwareAnalytics/RLCoder). We thank the authors of RLCoder, CrossCodeEval, RepoEval, UniXcoder, and the generator models used in our experiments.

Please cite the original projects when using their code, datasets, or checkpoints.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{TODO2026unifiedcontext,
  title   = {TODO: Paper Title},
  author  = {TODO: Author List},
  journal = {TODO: Venue or arXiv},
  year    = {2026}
}
```

Replace this entry with the final paper metadata after publication.

## License

See [LICENSE](LICENSE) for details. Components derived from third-party projects remain subject to their original licenses.
