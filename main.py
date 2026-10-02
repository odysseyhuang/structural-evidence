import os
import time
import json
import re
import torch
import random
import argparse
import numpy as np

from generator import Generator
from bm25 import TaskSpecificBM25
from retriever import Retriever, tokenize
from datasets import load_test_dataset, load_train_and_valid_dataset, construct_dataset, CodeBlock
from context_query import build_base_query, build_query_bundle
from context_candidates import (
    add_retrieval_trace_results,
    candidate_key,
    recall_multi_path_candidates,
    log_retrieval_trace,
    write_retrieval_trace,
)
from context_gate import apply_candidate_gate, apply_retrieved_context_gate
from context_graph import add_graph_trace_results, expand_context_graph_candidates

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from utils.eval_metric import compute_metric_stmt
from utils.eval_codereval import eval_codereval
from utils.model_utils import local_model_path
from prettytable import PrettyTable
import copy

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set seed
def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()


# Retrieves code blocks based on different inference types.
def retrieve_codeblocks(args, examples, bm25, retriever, dataset_name, is_training=False, inference_type=None):
    """
    Retrieves code blocks based on different inference types.
    :param args: An argument object containing configuration parameters.
    :param examples: Examples used for retrieval.
    :param bm25: An instance of the BM25 model.
    :param retriever: An instance of the retriever.
    :param dataset_name: The name of the dataset.
    :param is_training: Whether it is in training mode.
    :return: A list of retrieved code blocks.
    """
    if inference_type is None:
        inference_type = args.inference_type
    if inference_type in {"baseline", "zeroshot"}:
        return None, [[] for _ in range(len(examples))]

    bm25_topk, unixcoder_topk, context_len = 5, 5, 20
    if inference_type in ["bm25", "unixcoder", "unixcoder_with_rl"]:
        if dataset_name not in bm25:
            bm25[dataset_name] = TaskSpecificBM25(examples, args)

        if inference_type == "unixcoder":
            bm25_topk = 50 
        elif inference_type == "unixcoder_with_rl":
            bm25_topk = args.sample_number * 10 
            unixcoder_topk = args.sample_number 

        base_queries = [build_base_query(example, context_len=context_len) for example in examples]
        queries = base_queries
        ucm_trace_rows = None

        enable_ucm_multi_path = (
            getattr(args, "enable_ucm", False)
            and getattr(args, "enable_multi_path_retrieval", False)
        )
        enable_ucm_context_graph = (
            getattr(args, "enable_ucm", False)
            and getattr(args, "enable_context_graph", False)
        )

        if enable_ucm_multi_path:
            draft_generations = None
            if args.enable_repocoder and inference_type == 'unixcoder_with_rl':
                draft_args = copy.deepcopy(args)
                draft_args.enable_ucm = False
                draft_args.enable_multi_path_retrieval = False
                _, retrieved_codeblocks = retrieve_codeblocks(draft_args, examples, bm25, retriever_RLCoder, dataset_name, inference_type="unixcoder")
                draft_generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)
                queries = [query + '\n' + prediction for query, prediction in zip(base_queries, draft_generations)]

            query_bundles = [
                build_query_bundle(
                    args,
                    example,
                    base_query=base_query,
                    draft_prediction=draft_generations[idx] if draft_generations else None,
                    context_len=context_len,
                )
                for idx, (example, base_query) in enumerate(zip(examples, base_queries))
            ]
            candidate_codeblocks, ucm_trace_rows = recall_multi_path_candidates(
                args,
                examples,
                bm25[dataset_name],
                query_bundles,
                base_topk=bm25_topk,
            )
        else:
            candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in examples], queries, topk=bm25_topk)
            if enable_ucm_context_graph:
                ucm_trace_rows = _base_candidate_trace_rows(examples, candidate_codeblocks)

            if args.enable_repocoder and inference_type == 'unixcoder_with_rl':
                _, retrieved_codeblocks = retrieve_codeblocks(args, examples, bm25, retriever_RLCoder, dataset_name, inference_type="unixcoder")
                generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)

                queries = [query + '\n' + prediction for query, prediction in zip(queries, generations)]

        if enable_ucm_context_graph:
            candidate_codeblocks, graph_trace_rows = expand_context_graph_candidates(
                args,
                examples,
                bm25[dataset_name],
                candidate_codeblocks,
            )
            add_graph_trace_results(ucm_trace_rows, graph_trace_rows)

        if enable_ucm_multi_path:
            candidate_codeblocks = apply_candidate_gate(candidate_codeblocks, args)

        controller_candidate_pool = None
        if (
            getattr(args, "enable_evidence_controller", False)
            and getattr(args, "enable_ucm", False)
            and not is_training
        ):
            controller_candidate_pool = [list(blocks) for blocks in candidate_codeblocks]

        if inference_type == "bm25":
            _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
            if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
            return queries, candidate_codeblocks
        elif inference_type == "unixcoder":
            candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
            if not is_training:
                candidate_codeblocks = apply_retrieved_context_gate(candidate_codeblocks, args)
            _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
            if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
            return queries, candidate_codeblocks
        elif inference_type == "unixcoder_with_rl":
            if is_training:
                if args.disable_stop_block:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
                else:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk-1)

                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, examples)]
                _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
                if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                    log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
            else:
                if not args.disable_stop_block:
                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, examples)]

                candidate_codeblocks = retriever.retrieve(queries,  candidate_codeblocks, topk=unixcoder_topk)
                candidate_codeblocks = apply_retrieved_context_gate(candidate_codeblocks, args)
                retrieved_before_controller = [list(blocks) for blocks in candidate_codeblocks]
                if controller_candidate_pool is not None:
                    candidate_codeblocks, controller_trace_rows = controller_runtime.refine(
                        examples,
                        candidate_codeblocks,
                        controller_candidate_pool,
                    )
                    if args.controller_preserve_tail_evidence:
                        candidate_codeblocks, restored_counts = _restore_controller_tail_evidence(
                            candidate_codeblocks,
                            retrieved_before_controller,
                            args.controller_max_evidence_blocks,
                            args.controller_tail_evidence_limit or args.sample_number,
                        )
                        for trace_row, restored_count, final_blocks in zip(
                            controller_trace_rows, restored_counts, candidate_codeblocks
                        ):
                            trace_row["tail_evidence_restored"] = restored_count
                            trace_row["generator_evidence_count"] = len(final_blocks)
                    controller_runtime.write_trace(
                        args.output_dir,
                        dataset_name,
                        controller_trace_rows,
                    )
                elif args.final_evidence_limit:
                    candidate_codeblocks = _limit_final_evidence_blocks(
                        candidate_codeblocks, args.final_evidence_limit
                    )
                _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
                if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                    log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
        
            return queries, candidate_codeblocks

    raise ValueError("Unsupported inference type: {}".format(args.inference_type))


def _visible_evidence_blocks(blocks):
    visible = []
    for block in blocks:
        if not getattr(block, "file_path", ""):
            break
        visible.append(block)
    return visible


def _limit_final_evidence_blocks(retrieved_codeblocks, limit):
    if limit < 1:
        return retrieved_codeblocks
    return [
        _visible_evidence_blocks(blocks)[:limit]
        for blocks in retrieved_codeblocks
    ]


def _restore_controller_tail_evidence(
    controlled_codeblocks,
    original_codeblocks,
    controlled_prefix_size,
    final_limit,
):
    """Restore untouched retriever ranks after the Controller's trained prefix."""
    restored = []
    restored_counts = []
    for controlled, original in zip(controlled_codeblocks, original_codeblocks):
        merged = list(controlled)
        seen = {candidate_key(block) for block in merged}
        restored_count = 0
        for block in _visible_evidence_blocks(original)[controlled_prefix_size:]:
            key = candidate_key(block)
            if key in seen:
                continue
            if final_limit > 0 and len(merged) >= final_limit:
                break
            merged.append(block)
            seen.add(key)
            restored_count += 1
        restored.append(merged)
        restored_counts.append(restored_count)
    return restored, restored_counts


def _finalize_ucm_trace(args, dataset_name, trace_rows, retrieved_codeblocks):
    if not trace_rows:
        return

    add_retrieval_trace_results(trace_rows, retrieved_codeblocks)
    write_retrieval_trace(
        args.output_dir,
        dataset_name,
        trace_rows,
        overwrite=getattr(args, "ucm_trace_overwrite", False),
    )


def _apply_ucm_output_suffix(args):
    run_short_name = getattr(args, "run_short_name", "")
    if run_short_name:
        safe_name = _safe_run_short_name(run_short_name)
        output_dir = args.output_dir.rstrip("/\\")
        output_parent = os.path.dirname(output_dir) or "."
        args.output_dir = os.path.join(output_parent, safe_name)
        return

    if not (
        getattr(args, "enable_ucm", False)
        and getattr(args, "enable_multi_path_retrieval", False)
    ):
        return

    suffix = _ucm_output_suffix(args)
    output_dir = args.output_dir.rstrip("/\\")
    basename = os.path.basename(output_dir)
    if suffix not in basename:
        args.output_dir = f"{output_dir}_{suffix}"


def _safe_run_short_name(name):
    if not re.match(r"^[A-Za-z0-9._-]+$", name):
        raise ValueError("--run_short_name may only contain letters, numbers, dot, underscore, or hyphen")
    return name


def _write_run_config(args):
    os.makedirs(args.output_dir, exist_ok=True)
    graph_edges = []
    if getattr(args, "enable_context_graph", False):
        if not getattr(args, "ucm_graph_disable_same_file_edges", False):
            graph_edges.append("same_file")
        if getattr(args, "ucm_graph_enable_identifier_edges", False):
            graph_edges.append("identifier")
        if getattr(args, "ucm_graph_enable_import_edges", False):
            graph_edges.append("import_path")
        if getattr(args, "ucm_graph_enable_api_call_edges", False):
            graph_edges.append("api_call")
        if getattr(args, "ucm_graph_enable_typed_dependency_edges", False):
            graph_edges.append("typed_dependency")
        if getattr(args, "ucm_graph_enable_unified_context_edges", False):
            graph_edges.append("unified_context")

    query_views = ["base"]
    if getattr(args, "enable_multi_path_retrieval", False):
        if not getattr(args, "ucm_disable_identifier_query", False):
            query_views.append("identifier")
        if not getattr(args, "ucm_disable_import_api_query", False):
            query_views.append("import_api")
        if getattr(args, "ucm_enable_path_query", False):
            query_views.append("path")

    config = {
        "run_short_name": getattr(args, "run_short_name", ""),
        "output_dir": args.output_dir,
        "eval_datasets": getattr(args, "eval_datasets", ""),
        "queries": query_views,
        "ucm": {
            "enabled": getattr(args, "enable_ucm", False),
            "multi_path_retrieval": getattr(args, "enable_multi_path_retrieval", False),
            "base_topk": getattr(args, "ucm_base_topk", None),
            "topk_per_path": getattr(args, "ucm_topk_per_path", None),
            "path_topk": getattr(args, "ucm_path_topk", None),
            "candidate_pool_size": getattr(args, "ucm_candidate_pool_size", None),
            "enhanced_bm25": not getattr(args, "ucm_disable_enhanced_bm25", False),
            "context_gate": getattr(args, "enable_context_gate", False),
        },
        "controller": {
            "enabled": getattr(args, "enable_evidence_controller", False),
            "checkpoint": getattr(args, "controller_checkpoint", None),
            "encoder": getattr(args, "controller_encoder_model_path", None),
            "max_rounds": getattr(args, "controller_max_rounds", None),
            "max_evidence_blocks": getattr(args, "controller_max_evidence_blocks", None),
            "add_per_round": getattr(args, "controller_add_per_round", None),
            "conflict_action_threshold": getattr(
                args, "controller_conflict_action_threshold", None
            ),
            "remove_threshold": getattr(args, "controller_remove_threshold", None),
            "preserve_tail_evidence": getattr(
                args, "controller_preserve_tail_evidence", False
            ),
            "tail_evidence_limit": getattr(
                args, "controller_tail_evidence_limit", None
            ),
        },
        "final_evidence_limit": getattr(args, "final_evidence_limit", None),
        "graph": {
            "enabled": getattr(args, "enable_context_graph", False),
            "edges": graph_edges,
            "max_seed": getattr(args, "ucm_graph_max_seed", None),
            "neighbors_per_seed": getattr(args, "ucm_graph_max_neighbors_per_seed", None),
            "max_expanded": getattr(args, "ucm_graph_max_expanded", None),
            "same_file_direction": getattr(args, "ucm_graph_same_file_direction", None),
            "identifier_query_only": getattr(args, "ucm_graph_identifier_query_only", False),
            "identifier_max_df": getattr(args, "ucm_graph_identifier_max_df", None),
            "seed_rank_decay": getattr(args, "ucm_graph_seed_rank_decay", None),
            "distance_decay": getattr(args, "ucm_graph_distance_decay", None),
            "same_file_weight": getattr(args, "ucm_graph_same_file_weight", None),
            "identifier_weight": getattr(args, "ucm_graph_identifier_weight", None),
            "import_weight": getattr(args, "ucm_graph_import_weight", None),
            "api_call_weight": getattr(args, "ucm_graph_api_call_weight", None),
            "api_call_max_df": getattr(args, "ucm_graph_api_call_max_df", None),
            "api_call_query_only": getattr(args, "ucm_graph_api_call_query_only", False),
            "query_overlap_bonus": getattr(args, "ucm_graph_query_overlap_bonus", None),
            "query_api_bonus": getattr(args, "ucm_graph_query_api_bonus", None),
            "max_selected": getattr(args, "ucm_graph_max_selected", None),
            "max_selected_tokens": getattr(
                args, "ucm_graph_max_selected_tokens", None
            ),
            "show_relations_in_prompt": getattr(
                args, "ucm_graph_show_relations_in_prompt", False
            ),
            "show_high_confidence_paths": getattr(
                args, "ucm_graph_show_high_confidence_paths", False
            ),
            "prompt_min_confidence": getattr(
                args, "ucm_graph_prompt_min_confidence", None
            ),
            "relation_max_symbols": getattr(
                args, "ucm_graph_relation_max_symbols", None
            ),
            "typed_dependency": {
                "enabled": getattr(
                    args, "ucm_graph_enable_typed_dependency_edges", False
                ),
                "extractor": "ast_regex_v1",
                "relation_mode": getattr(
                    args, "ucm_graph_typed_relation_mode", "all"
                ),
                "query_max": getattr(args, "ucm_graph_typed_query_max", None),
                "query_context_lines": getattr(
                    args, "ucm_graph_typed_query_context_lines", None
                ),
                "max_df": getattr(args, "ucm_graph_typed_max_df", None),
                "query_bonus": getattr(
                    args, "ucm_graph_typed_query_bonus", None
                ),
                "call_weight": getattr(
                    args, "ucm_graph_typed_call_weight", None
                ),
                "type_weight": getattr(
                    args, "ucm_graph_typed_type_weight", None
                ),
                "def_use_weight": getattr(
                    args, "ucm_graph_typed_def_use_weight", None
                ),
                "allow_target_file": getattr(
                    args, "ucm_graph_typed_allow_target_file", False
                ),
            },
            "unified_context": {
                "enabled": getattr(
                    args, "ucm_graph_enable_unified_context_edges", False
                ),
                "multi_evidence": getattr(
                    args, "ucm_graph_enable_multi_evidence", False
                ),
                "query_max": getattr(args, "ucm_graph_unified_query_max", None),
                "query_context_lines": getattr(
                    args, "ucm_graph_unified_query_context_lines", None
                ),
                "max_df": getattr(args, "ucm_graph_unified_max_df", None),
                "max_evidence_per_candidate": getattr(
                    args, "ucm_graph_max_evidence_per_candidate", None
                ),
                "receiver_weight": getattr(
                    args, "ucm_graph_unified_receiver_weight", None
                ),
                "override_weight": getattr(
                    args, "ucm_graph_unified_override_weight", None
                ),
                "member_weight": getattr(
                    args, "ucm_graph_unified_member_weight", None
                ),
                "import_weight": getattr(
                    args, "ucm_graph_unified_import_weight", None
                ),
                "inheritance_weight": getattr(
                    args, "ucm_graph_unified_inheritance_weight", None
                ),
                "signature_weight": getattr(
                    args, "ucm_graph_unified_signature_weight", None
                ),
                "control_weight": getattr(
                    args, "ucm_graph_unified_control_weight", None
                ),
            },
            "rerank": {
                "enabled": getattr(args, "ucm_graph_enable_rerank", False),
                "alpha": getattr(args, "ucm_graph_rerank_alpha", None),
                "source_prior_same_file": getattr(args, "ucm_graph_source_prior_same_file", None),
                "source_prior_identifier": getattr(args, "ucm_graph_source_prior_identifier", None),
                "source_prior_import": getattr(args, "ucm_graph_source_prior_import", None),
                "source_prior_api_call": getattr(args, "ucm_graph_source_prior_api_call", None),
                "distance_penalty": getattr(args, "ucm_graph_distance_penalty", None),
                "path_rerank": getattr(
                    args, "ucm_graph_enable_path_rerank", False
                ),
                "path_score_scale": getattr(
                    args, "ucm_graph_path_score_scale", None
                ),
                "multi_relation_bonus": getattr(
                    args, "ucm_graph_path_multi_relation_bonus", None
                ),
                "query_origin_bonus": getattr(
                    args, "ucm_graph_path_query_origin_bonus", None
                ),
                "evidence_gate": getattr(
                    args, "ucm_graph_enable_evidence_gate", False
                ),
                "new_candidate_min_paths": getattr(
                    args, "ucm_graph_new_candidate_min_paths", None
                ),
                "new_candidate_min_confidence": getattr(
                    args, "ucm_graph_new_candidate_min_confidence", None
                ),
            },
        },
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def _base_candidate_trace_rows(examples, candidate_codeblocks):
    trace_rows = []
    for example, candidates in zip(examples, candidate_codeblocks):
        candidate_count = len(candidates)
        trace_rows.append(
            {
                "task_id": example.task_id,
                "source_hits": {"base": candidate_count},
                "raw_hits": candidate_count,
                "merged_candidates": candidate_count,
                "candidate_pool_size": candidate_count,
            }
        )
    return trace_rows


def _ucm_output_suffix(args):
    parts = [
        "noid" if getattr(args, "ucm_disable_identifier_query", False) else "id",
        "noimport" if getattr(args, "ucm_disable_import_api_query", False) else "import",
        "path" if getattr(args, "ucm_enable_path_query", False) else "nopath",
        f"base{getattr(args, 'ucm_base_topk', 60)}",
        f"aux{getattr(args, 'ucm_topk_per_path', 30)}",
        f"pathk{getattr(args, 'ucm_path_topk', 5)}",
        f"pool{getattr(args, 'ucm_candidate_pool_size', 140)}",
        "legacybm25" if getattr(args, "ucm_disable_enhanced_bm25", False) else "enhbm25",
    ]
    if getattr(args, "enable_context_gate", False):
        parts.append(
            "gate_"
            f"aux{getattr(args, 'ucm_gate_max_auxiliary_blocks', 2)}_"
            f"path{getattr(args, 'ucm_gate_allow_path_only', 0)}_"
            f"stop{getattr(args, 'ucm_gate_stop_rank_threshold', 2)}"
        )
    else:
        parts.append("nogate")
    if getattr(args, "enable_context_graph", False):
        graph_parts = [
            f"graph_s{getattr(args, 'ucm_graph_max_seed', 20)}",
            f"n{getattr(args, 'ucm_graph_max_neighbors_per_seed', 2)}",
            f"e{getattr(args, 'ucm_graph_max_expanded', 40)}",
            "scored",
        ]
        if getattr(args, "ucm_graph_enable_identifier_edges", False):
            graph_parts.append("gid")
        if getattr(args, "ucm_graph_enable_import_edges", False):
            graph_parts.append("gimport")
        if getattr(args, "ucm_graph_enable_api_call_edges", False):
            graph_parts.append("gapi")
        if getattr(args, "ucm_graph_enable_typed_dependency_edges", False):
            graph_parts.append("gtyped")
            relation_mode = getattr(args, "ucm_graph_typed_relation_mode", "all")
            if relation_mode != "all":
                graph_parts.append(relation_mode)
        if getattr(args, "ucm_graph_enable_unified_context_edges", False):
            graph_parts.append("gunified")
        if getattr(args, "ucm_graph_enable_multi_evidence", False):
            graph_parts.append("multiev")
        max_selected = getattr(args, "ucm_graph_max_selected", 0)
        if max_selected:
            graph_parts.append(f"gcap{max_selected}")
        if getattr(args, "ucm_graph_show_relations_in_prompt", False):
            graph_parts.append("gvisible")
        if getattr(args, "ucm_graph_show_high_confidence_paths", False):
            graph_parts.append("pathvisible")
        if getattr(args, "ucm_graph_enable_rerank", False):
            graph_parts.append(
                "rerank"
                f"a{getattr(args, 'ucm_graph_rerank_alpha', 0.03)}"
                f"d{getattr(args, 'ucm_graph_distance_penalty', 0.005)}"
            )
        parts.append("_".join(graph_parts))
    return "_".join(parts)


class CustomDataset(Dataset):
    def __init__(self, max_query_length, max_candidate_length, tokenizer, queries, candidates, labels):
        self.max_query_length = max_query_length
        self.max_candidate_length = max_candidate_length
        self.tokenizer = tokenizer
        self.queries = queries
        self.candidates = candidates
        self.labels = labels

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_tokens_id = tokenize(self.queries[idx], self.tokenizer, self.max_query_length, True)
        candidate_tokens_id = [tokenize(str(x), self.tokenizer, self.max_candidate_length, False) for x in self.candidates[idx]]
        return torch.tensor(query_tokens_id, dtype=torch.long), torch.tensor(candidate_tokens_id, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

EVAL_DATASET_ORDER = (
    "github_eval",
    "cceval_python",
    "cceval_java",
    "repoeval_line",
    "repoeval_api",
)


def _load_selected_eval_examples(args):
    requested = [
        name.strip()
        for name in getattr(args, "eval_datasets", "").split(",")
        if name.strip()
    ]
    selected = requested or list(EVAL_DATASET_ORDER)
    unknown = sorted(set(selected) - set(EVAL_DATASET_ORDER))
    if unknown:
        raise ValueError(
            "Unknown --eval_datasets values: {}. Choose from: {}".format(
                ", ".join(unknown), ", ".join(EVAL_DATASET_ORDER)
            )
        )

    all_eval_examples = {}
    training_raw_data = None
    for name in EVAL_DATASET_ORDER:
        if name not in selected:
            continue
        if name == "github_eval":
            training_raw_data, eval_raw_data = load_train_and_valid_dataset()
            all_eval_examples[name] = construct_dataset(
                eval_raw_data, 100 if args.debug else 1000
            )
        elif name == "cceval_python":
            all_eval_examples[name] = load_test_dataset(args, "cceval", "python")
        elif name == "cceval_java":
            all_eval_examples[name] = load_test_dataset(args, "cceval", "java")
        elif name == "repoeval_line":
            all_eval_examples[name] = load_test_dataset(
                args, "repoeval", "line_level"
            )
        elif name == "repoeval_api":
            all_eval_examples[name] = load_test_dataset(
                args, "repoeval", "api_level"
            )
    return all_eval_examples, training_raw_data


def run(args):
    if args.inference_type == "zeroshot" and not args.eval:
        raise ValueError("ZeroShot is an evaluation-only inference mode")

    all_eval_examples, training_raw_data = _load_selected_eval_examples(args)
    if not args.eval and training_raw_data is None:
        training_raw_data, _ = load_train_and_valid_dataset()


    global generator
    generator = Generator(args)
    # A pure ZeroShot run must not load or execute any retrieval model.
    retriever = None if args.inference_type == "zeroshot" else Retriever(args)

    global controller_runtime
    controller_runtime = None
    if args.enable_evidence_controller:
        if not args.eval:
            raise ValueError("The frozen evidence Controller is available only during evaluation")
        if args.inference_type != "unixcoder_with_rl" or not args.enable_ucm:
            raise ValueError("The evidence Controller requires UCM unixcoder_with_rl retrieval")
        from evidence_controller.runtime import ControllerRuntime

        controller_runtime = ControllerRuntime(
            checkpoint_path=args.controller_checkpoint,
            encoder_model_path=args.controller_encoder_model_path,
            device="cuda",
            batch_size=args.controller_batch_size,
            max_rounds=args.controller_max_rounds,
            max_evidence_blocks=args.controller_max_evidence_blocks,
            add_per_round=args.controller_add_per_round,
            conflict_action_threshold=args.controller_conflict_action_threshold,
            remove_threshold=args.controller_remove_threshold,
        )


    if args.enable_repocoder:
        args_RLCoder = copy.deepcopy(args)
        args_RLCoder.retriever_model_path = args.rlcoder_model_path
        global retriever_RLCoder
        retriever_RLCoder = Retriever(args_RLCoder)
    

    if not args.enable_forward_generation:
        args.forward_generation_times = 1
    else:
        if args.forward_generation_times is None:
            args.forward_generation_times = 4

    bm25 = {}
    
    if args.eval:
        table = PrettyTable()
        table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        codereval_table = PrettyTable()
        codereval_table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
        
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            print("Evaluating on {} dataset".format(name))
            
            temp_examples = copy.deepcopy(examples)
            temp_generations = []
                
            for _ in range(args.forward_generation_times):
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                # for i in range(len(retrieved_codeblocks)):
                #     for j in range(len(retrieved_codeblocks[i])):
                #         print('#', retrieved_codeblocks[i][j].file_path)
                #         print(retrieved_codeblocks[i][j].code_content)
                losses = generator.evaluate(examples, retrieved_codeblocks)


                results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if not os.path.exists(f"{args.output_dir}/{name}"):
                    os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
                with open(f"{args.output_dir}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")


                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, temp_generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in temp_generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(temp_generations,targets)])/len(temp_generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                # elif name == "repoeval_func":
                #     results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/func_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
            

            if 'codereval' in name:
                codereval_table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(table)
            print(codereval_table)
        
    else:
        print("data_per_epoch:{}, batch_size:{}, sample_number:{}, epoch:{}, inner_epoch:{}, lr:{}".format(args.data_per_epoch, args.batch_size,args.sample_number,args.epoch,args.inner_epoch,args.lr))
        optimizer = AdamW(retriever.model.parameters(), lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.2, num_training_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch)
    
        evaluate_table = {}
        for name, examples in all_eval_examples.items():
            evaluate_table[name] = PrettyTable()
            if 'codereval' in name:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
            else:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        training_table = PrettyTable()
        training_table.field_names = ["Epoch", "Dataset", "Total Samples", "Rewards", "Training Loss", "Time (sec)"]


        retriever.model.eval()
        for name, examples in all_eval_examples.items():
            # examples = examples[:10]
            
            start_time = time.time()
            temp_examples = copy.deepcopy(examples)
            temp_generations = []

                
            for _ in range(args.forward_generation_times):
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                losses = generator.evaluate(examples, retrieved_codeblocks)


                results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if os.path.exists(f"{args.output_dir}/result_init/{name}") is False:
                    os.makedirs(f"{args.output_dir}/result_init/{name}", exist_ok=True)
                with open(f"{args.output_dir}/result_init/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")

                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations, targets)])/len(generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                

            if 'codereval' in name:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(evaluate_table[name])


        for epoch in range(args.epoch):
            print("=" * 40 + "Epoch:{}".format(epoch) + "=" * 40)
            retriever.model.eval()
            start_time = time.time()
            results = {}
            results["Epoch"] = epoch


            training_examples = construct_dataset(training_raw_data, 100 if args.debug else args.data_per_epoch)
            # training_examples = construct_dataset(training_raw_data, 100)
            queries, retrieved_codeblocks = retrieve_codeblocks(args, training_examples, bm25, retriever, "github_training_{}".format(epoch), True)
            training_examples_dup = [x for x in training_examples for _ in range(args.sample_number)]
            training_codeblocks_dup = [[x] for y in retrieved_codeblocks for x in y]
            assert len(training_examples_dup) == len(training_codeblocks_dup)


            losses = generator.evaluate(training_examples_dup, training_codeblocks_dup)
            labels = torch.tensor([x for x in losses]).view(-1, args.sample_number).argmin(-1)
            results["Total Samples"] = len(queries)
            results["Rewards"] = labels.float().mean().item()

            retriever.model.train()
            total_loss = 0
            dataset = CustomDataset(args.retriever_query_context_length, args.retriever_candidate_context_length, retriever.tokenizer, queries, retrieved_codeblocks, labels.tolist())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            for inner_epoch in range(args.inner_epoch):
                for batch in dataloader:
                    source_ids, doc_ids, labels = [x.cuda() for x in batch]
                    queries_embeddings = retriever(source_ids)
                    doc_texts_embeddings = retriever(doc_ids.view(-1, doc_ids.shape[-1])).view(source_ids.shape[0], args.sample_number, -1)
                    logits = torch.einsum("ab,acb->ac", queries_embeddings, doc_texts_embeddings)*20
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(retriever.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    total_loss += loss.item()

                if args.enable_sft:
                    retriever.model.eval()
                    for name, examples in all_eval_examples.items():
                        # examples = examples[:10]
                        
                        start_time = time.time()
                        temp_examples = copy.deepcopy(examples)
                        temp_generations = []

                            
                        for _ in range(args.forward_generation_times):
                            _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                            losses = generator.evaluate(examples, retrieved_codeblocks)

                            results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                            if args.enable_generation:
                                generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                                if not temp_generations:
                                    temp_generations = generations
                                else:
                                    temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                                for i in range(len(temp_examples)):
                                    temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                                    
                        if args.enable_generation:
                            if os.path.exists(f"{args.output_dir}/result_{inner_epoch}/{name}") is False:
                                os.makedirs(f"{args.output_dir}/result_{inner_epoch}/{name}", exist_ok=True)
                            with open(f"{args.output_dir}/result_{inner_epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                                for example, generation in zip(examples, temp_generations):
                                    f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                            if name == "cceval_python":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "cceval_java":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                            elif name == "github_eval":
                                targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                                results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                            elif name == "codereval_python":
                                results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                            elif name == "codereval_java":
                                results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                            elif name == "repoeval_line":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "repoeval_api":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                        if 'codereval' in name:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                        else:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                        print(evaluate_table[name])
                    
                    retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")
                    retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")

            results["Training Loss"] = total_loss/len(dataloader)/args.inner_epoch
            results["Time (sec)"] = round(time.time() - start_time, 1)
            training_table.add_row([results["Epoch"], "github_training_{}".format(epoch), results["Total Samples"], results["Rewards"], results["Training Loss"], results["Time (sec)"]])
            print(training_table)

            
            retriever.model.eval()
            for name, examples in all_eval_examples.items():
                # examples = examples[:10]
                
                start_time = time.time()
                temp_examples = copy.deepcopy(examples)
                temp_generations = []
                    
                for _ in range(args.forward_generation_times):
                    _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                    losses = generator.evaluate(examples, retrieved_codeblocks)

                    results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                    if args.enable_generation:
                        generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                        if not temp_generations:
                            temp_generations = generations
                        else:
                            temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                        for i in range(len(temp_examples)):
                            temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                            
                if args.enable_generation:
                    if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
                        os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
                    with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                        for example, generation in zip(examples, temp_generations):
                            f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                    if name == "cceval_python":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "cceval_java":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                    elif name == "github_eval":
                        targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                        results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                    elif name == "codereval_python":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                    elif name == "codereval_java":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                    elif name == "repoeval_line":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "repoeval_api":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                if 'codereval' in name:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                else:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                print(evaluate_table[name])

            retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_model_path", default=local_model_path("deepseek-coder-6.7b-base"), type=str, help="Generator model path")
    parser.add_argument("--generator_batch_size_per_gpu", default=32, type=int, help="Generator batch size per GPU")
    parser.add_argument("--generator_max_crossfile_length", default=3072, type=int, help="Maximum cross-file length for the generator")
    parser.add_argument("--generator_max_context_length", default=4096, type=int, help="Maximum context length for the generator")
    parser.add_argument("--generator_max_generation_length", default=64, type=int, help="Maximum generation length for the generator")
    parser.add_argument("--disable_generator", action="store_true", help="Disable the generator")

    parser.add_argument("--retriever_model_path", default=local_model_path("unixcoder-base"), type=str, help="Retriever model path")
    parser.add_argument("--retriever_batch_size_per_gpu", default=64, type=int, help="Retriever batch size per GPU")
    parser.add_argument("--disable_retriever", action="store_true", help="Disable the retriever")
    parser.add_argument("--retriever_query_context_length", default=256, type=int, help="Retriever query context length")
    parser.add_argument("--retriever_candidate_context_length", default=512, type=int, help="Retriever candidate context length")

    parser.add_argument(
        "--inference_type",
        default="baseline",
        type=str,
        help="Inference type; use 'zeroshot' for generator-only completion without retrieval or prompt metadata",
    )
    parser.add_argument("--output_dir", default="results/baseline", type=str, help="Output directory")
    parser.add_argument("--run_short_name", default="", type=str, help="Optional short result directory name under the output parent directory")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation")
    parser.add_argument("--eval_datasets", default="", type=str, help="Optional comma-separated evaluation dataset names")
    parser.add_argument("--enable_tqdm", action="store_true", help="Enable progress bar")
    parser.add_argument("--enable_generation", action="store_true", help="Enable generation")
    parser.add_argument("--debug", action="store_true", help="Debug mode, use a small dataset")

    parser.add_argument("--num_workers", default=14, type=int, help="Number of CPU cores")
    parser.add_argument("--weighted_keywords", action="store_true", help="Weight keywords when calculating loss during training")
    parser.add_argument("--enable_fixed_block", action="store_true", help="Use fixed length blocks when building candidates")
    parser.add_argument("--enable_sft", action="store_true", help="Train using supervised learning methods")
    parser.add_argument("--disable_stop_block", action="store_true", help="Disable the stop block")

    parser.add_argument("--enable_repocoder", action="store_true", help="Use the repocoder method during generation")
    parser.add_argument("--rlcoder_model_path", default=local_model_path("unixcoder-base"), type=str, help="Stage 1 model for repocoder")

    parser.add_argument("--enable_ucm", action="store_true", help="Enable UCM extensions")
    parser.add_argument("--enable_multi_path_retrieval", action="store_true", help="Enable UCM multi-path BM25 candidate recall")
    parser.add_argument("--ucm_base_topk", default=60, type=int, help="BM25 candidates for the base query; 0 keeps the original RLCoder topK")
    parser.add_argument("--ucm_topk_per_path", default=30, type=int, help="Number of BM25 candidates per auxiliary UCM query view")
    parser.add_argument("--ucm_path_topk", default=5, type=int, help="Number of BM25 candidates for the optional path query view")
    parser.add_argument("--ucm_candidate_pool_size", default=140, type=int, help="Maximum merged UCM candidate pool size")
    parser.add_argument("--ucm_disable_identifier_query", action="store_true", help="Disable the UCM identifier query view")
    parser.add_argument("--ucm_disable_import_api_query", action="store_true", help="Disable the UCM import/API query view")
    parser.add_argument("--ucm_enable_path_query", action="store_true", help="Enable the path-based UCM query view")
    parser.add_argument("--ucm_query_identifier_limit", default=64, type=int, help="Maximum identifier tokens in the UCM identifier query")
    parser.add_argument("--ucm_query_import_limit", default=32, type=int, help="Maximum import/API lines or tokens in the UCM import query")
    parser.add_argument("--ucm_disable_enhanced_bm25", action="store_true", help="Use the original BM25 tokenization and code-only index for UCM")
    parser.add_argument("--ucm_trace_retrieval", action="store_true", help="Print UCM retrieval trace; jsonl trace files are always written for UCM retrieval")
    parser.add_argument("--ucm_trace_overwrite", action="store_true", help="Overwrite each dataset trace instead of appending to an existing result directory")
    parser.add_argument("--enable_context_gate", action="store_true", help="Enable lightweight rule-based UCM context gate")
    parser.add_argument("--ucm_gate_max_auxiliary_blocks", default=2, type=int, help="Maximum auxiliary-only UCM candidates kept before reranking")
    parser.add_argument("--ucm_gate_allow_path_only", default=0, type=int, help="Maximum path-only UCM candidates kept before reranking")
    parser.add_argument("--ucm_gate_stop_rank_threshold", default=2, type=int, help="Trim final context after an early stop block at or before this rank")
    parser.add_argument("--enable_context_graph", action="store_true", help="Enable lightweight UCM context graph expansion before RLRetriever reranking")
    parser.add_argument("--ucm_graph_max_seed", default=20, type=int, help="Maximum retrieved/merged seed candidates used for graph expansion")
    parser.add_argument("--ucm_graph_max_neighbors_per_seed", default=2, type=int, help="Maximum graph neighbors added for each seed candidate")
    parser.add_argument("--ucm_graph_max_expanded", default=40, type=int, help="Maximum total graph-expanded candidates per example")
    parser.add_argument("--ucm_graph_disable_same_file_edges", action="store_true", help="Disable positional same-file graph edges")
    parser.add_argument("--ucm_graph_enable_identifier_edges", action="store_true", help="Enable identifier-overlap graph edges")
    parser.add_argument("--ucm_graph_enable_import_edges", action="store_true", help="Enable import/path graph edges")
    parser.add_argument("--ucm_graph_enable_api_call_edges", action="store_true", help="Enable API/call-name graph edges")
    parser.add_argument("--ucm_graph_enable_typed_dependency_edges", action="store_true", help="Enable DDG-lite typed dependency edges backed by code definitions")
    parser.add_argument("--ucm_graph_enable_unified_context_edges", action="store_true", help="Enable the full visible-prefix and repository heterogeneous context graph")
    parser.add_argument("--ucm_graph_enable_multi_evidence", action="store_true", help="Preserve and aggregate all graph paths supporting each candidate")
    parser.add_argument("--ucm_graph_typed_relation_mode", default="all", choices=["all", "type_only", "call_only", "def_use_only"], help="Typed dependency relations enabled during graph expansion")
    parser.add_argument("--ucm_graph_identifier_max_df", default=20, type=int, help="Maximum per-task document frequency for identifier graph edges")
    parser.add_argument("--ucm_graph_api_call_max_df", default=20, type=int, help="Maximum per-task document frequency for API/call graph edges")
    parser.add_argument("--ucm_graph_same_file_direction", default="both", choices=["both", "prev", "next"], help="Same-file graph neighbor direction")
    parser.add_argument("--ucm_graph_identifier_query_only", action="store_true", help="Use only query-side identifiers for identifier graph expansion")
    parser.add_argument("--ucm_graph_api_call_query_only", action="store_true", help="Use only query-side API/call tokens for API graph expansion")
    parser.add_argument("--ucm_graph_seed_rank_decay", default=0.05, type=float, help="Decay applied to graph expansion candidates from lower-ranked seed blocks")
    parser.add_argument("--ucm_graph_distance_decay", default=0.75, type=float, help="Distance decay for same-file graph neighbors")
    parser.add_argument("--ucm_graph_same_file_weight", default=1.0, type=float, help="Base score weight for same-file graph edges")
    parser.add_argument("--ucm_graph_identifier_weight", default=1.2, type=float, help="Base score weight for identifier-overlap graph edges")
    parser.add_argument("--ucm_graph_import_weight", default=1.4, type=float, help="Base score weight for import/path graph edges")
    parser.add_argument("--ucm_graph_api_call_weight", default=1.6, type=float, help="Base score weight for API/call-name graph edges")
    parser.add_argument("--ucm_graph_query_overlap_bonus", default=2, type=int, help="Extra identifier graph score for tokens also present in the query context")
    parser.add_argument("--ucm_graph_query_api_bonus", default=2, type=int, help="Extra API/call graph score for tokens also present in the query context")
    parser.add_argument("--ucm_graph_typed_query_max", default=8, type=int, help="Maximum direct query-to-definition typed dependency candidates")
    parser.add_argument("--ucm_graph_typed_query_context_lines", default=80, type=int, help="Recent left-context lines used to build direct typed dependency requests")
    parser.add_argument("--ucm_graph_typed_max_df", default=12, type=int, help="Maximum number of definition blocks allowed for a typed dependency symbol")
    parser.add_argument("--ucm_graph_typed_query_bonus", default=1.0, type=float, help="Score bonus for typed dependencies originating directly from the query")
    parser.add_argument("--ucm_graph_typed_call_weight", default=1.8, type=float, help="Base graph score for call-to-definition dependencies")
    parser.add_argument("--ucm_graph_typed_type_weight", default=2.0, type=float, help="Base graph score for type-to-definition dependencies")
    parser.add_argument("--ucm_graph_typed_def_use_weight", default=1.4, type=float, help="Base graph score for use-to-definition dependencies")
    parser.add_argument("--ucm_graph_typed_allow_target_file", action="store_true", help="Allow typed dependency expansion into the target file; disabled by default to prevent leakage")
    parser.add_argument("--ucm_graph_unified_query_max", default=16, type=int, help="Maximum direct query-anchor candidates from the unified graph")
    parser.add_argument("--ucm_graph_unified_query_context_lines", default=160, type=int, help="Visible left-context lines used to build the local query graph")
    parser.add_argument("--ucm_graph_unified_max_df", default=12, type=int, help="Maximum definition ambiguity accepted by unified semantic paths")
    parser.add_argument("--ucm_graph_max_evidence_per_candidate", default=12, type=int, help="Maximum distinct graph evidence paths retained per candidate")
    parser.add_argument("--ucm_graph_unified_receiver_weight", default=3.2, type=float, help="Weight for reaching-type to receiver-member paths")
    parser.add_argument("--ucm_graph_unified_override_weight", default=2.9, type=float, help="Weight for inheritance-aware override paths")
    parser.add_argument("--ucm_graph_unified_member_weight", default=2.6, type=float, help="Weight for member-of paths")
    parser.add_argument("--ucm_graph_unified_import_weight", default=2.5, type=float, help="Weight for import and package resolution paths")
    parser.add_argument("--ucm_graph_unified_inheritance_weight", default=2.3, type=float, help="Weight for extends and implements paths")
    parser.add_argument("--ucm_graph_unified_signature_weight", default=1.7, type=float, help="Weight for parameter and expected-return type paths")
    parser.add_argument("--ucm_graph_unified_control_weight", default=0.8, type=float, help="Weight for visible-prefix control-dependence evidence")
    parser.add_argument("--ucm_graph_max_selected", default=0, type=int, help="Maximum graph-only blocks allowed in final Top-K; 0 disables the cap")
    parser.add_argument("--ucm_graph_max_selected_tokens", default=0, type=int, help="Approximate retriever-token budget for graph-only blocks in final Top-K; 0 disables the cap")
    parser.add_argument("--ucm_graph_show_relations_in_prompt", action="store_true", help="Expose typed dependency relation headers to the generator only")
    parser.add_argument("--ucm_graph_show_high_confidence_paths", action="store_true", help="Expose one high-confidence path for graph-only generator context")
    parser.add_argument("--ucm_graph_prompt_min_confidence", default=3.0, type=float, help="Minimum path confidence exposed to the generator")
    parser.add_argument("--ucm_graph_relation_max_symbols", default=3, type=int, help="Maximum matched symbols shown in each dependency relation header")
    parser.add_argument("--ucm_graph_enable_rerank", action="store_true", help="Enable graph-aware score fusion after RLRetriever cosine scoring")
    parser.add_argument("--ucm_graph_enable_path_rerank", action="store_true", help="Use absolute multi-path evidence instead of per-query min-max graph scores")
    parser.add_argument("--ucm_graph_path_score_scale", default=4.0, type=float, help="Scale for bounded unified path-score reranking")
    parser.add_argument("--ucm_graph_path_multi_relation_bonus", default=0.004, type=float, help="Rerank bonus per additional supporting relation")
    parser.add_argument("--ucm_graph_path_query_origin_bonus", default=0.003, type=float, help="Rerank bonus for direct query-anchor evidence")
    parser.add_argument("--ucm_graph_enable_evidence_gate", action="store_true", help="Require multiple or high-confidence paths for graph-only context")
    parser.add_argument("--ucm_graph_new_candidate_min_paths", default=2, type=int, help="Minimum evidence paths for a graph-only candidate")
    parser.add_argument("--ucm_graph_new_candidate_min_confidence", default=3.0, type=float, help="Single-path confidence that bypasses the graph-only path-count gate")
    parser.add_argument("--ucm_graph_rerank_alpha", default=0.03, type=float, help="Weight for normalized graph score in graph-aware reranking")
    parser.add_argument("--ucm_graph_source_prior_same_file", default=0.02, type=float, help="Rerank prior for same-file graph candidates")
    parser.add_argument("--ucm_graph_source_prior_identifier", default=0.015, type=float, help="Rerank prior for identifier graph candidates")
    parser.add_argument("--ucm_graph_source_prior_import", default=0.005, type=float, help="Rerank prior for import/path graph candidates")
    parser.add_argument("--ucm_graph_source_prior_api_call", default=0.025, type=float, help="Rerank prior for API/call graph candidates")
    parser.add_argument("--ucm_graph_distance_penalty", default=0.005, type=float, help="Distance penalty for graph-aware reranking")

    parser.add_argument("--enable_evidence_controller", action="store_true", help="Enable the frozen Work-II evidence Controller and bounded retrieval policy")
    parser.add_argument("--controller_checkpoint", default="checkpoints/work2_controller_v1/best_controller.pt", type=str, help="Trained evidence Controller checkpoint")
    parser.add_argument("--controller_encoder_model_path", default=local_model_path("unixcoder-base"), type=str, help="Frozen UniXcoder model used by the Controller")
    parser.add_argument("--controller_batch_size", default=64, type=int, help="Controller and UniXcoder inference batch size")
    parser.add_argument("--controller_max_rounds", default=2, type=int, help="Maximum Controller retrieval feedback rounds")
    parser.add_argument("--controller_max_evidence_blocks", default=6, type=int, help="Maximum evidence blocks assessed and passed to generation")
    parser.add_argument("--controller_add_per_round", default=2, type=int, help="Maximum directed additions per feedback round")
    parser.add_argument("--controller_conflict_action_threshold", default=0.70, type=float, help="Conservative conflict probability required before removing evidence")
    parser.add_argument("--controller_remove_threshold", default=0.35, type=float, help="Maximum relative confidence for removing a conflicting candidate")
    parser.add_argument("--final_evidence_limit", default=0, type=int, help="Optional post-retrieval evidence limit for no-Controller ablations; 0 keeps the original list")
    parser.add_argument("--controller_preserve_tail_evidence", action="store_true", help="Restore untouched ranks after the Controller's trained evidence prefix before generation")
    parser.add_argument("--controller_tail_evidence_limit", default=0, type=int, help="Final evidence limit for preserved Controller tail; 0 uses --sample_number")

    parser.add_argument("--do_codereval", action="store_true", help="Execute codereval evaluation in docker")
    parser.add_argument("--enable_forward_generation", action="store_true", help="Use progressive generation methods during inference")
    parser.add_argument("--forward_generation_times", default=4, type=int, help="Number of times for progressive generation")

    parser.add_argument("--epoch", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--inner_epoch", default=1, type=int, help="Number of inner training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--sample_number", default=10, type=int, help="Number of samples")
    parser.add_argument("--data_per_epoch", default=2000, type=int, help="Amount of data per epoch")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")


    print("Number of GPUs:", torch.cuda.device_count())

    args = parser.parse_args()
    if args.final_evidence_limit < 0 or args.controller_tail_evidence_limit < 0:
        parser.error("evidence limits must be non-negative")
    if args.controller_preserve_tail_evidence and not args.enable_evidence_controller:
        parser.error("--controller_preserve_tail_evidence requires --enable_evidence_controller")
    if (
        args.controller_preserve_tail_evidence
        and args.controller_tail_evidence_limit
        and args.controller_tail_evidence_limit <= args.controller_max_evidence_blocks
    ):
        parser.error("--controller_tail_evidence_limit must exceed --controller_max_evidence_blocks")
    _apply_ucm_output_suffix(args)
    print("Output dir:", args.output_dir)
    _write_run_config(args)
    args.generator_batch_size = args.generator_batch_size_per_gpu * torch.cuda.device_count()
    args.retriever_batch_size = args.retriever_batch_size_per_gpu * torch.cuda.device_count()

    run(args)
