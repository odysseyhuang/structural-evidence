import json
import os
from collections import Counter

from datasets import CodeBlock


SOURCE_PRIORITY = {
    "base": 0,
    "draft": 1,
    "identifier": 2,
    "import_api": 3,
    "path": 4,
}


def recall_multi_path_candidates(args, examples, bm25_index, query_bundles, base_topk=None):
    per_task_hits = [[] for _ in examples]
    source_counts = [Counter() for _ in examples]

    for source in _ordered_sources(query_bundles):
        batch_indices = []
        task_ids = []
        queries = []

        for idx, (example, bundle) in enumerate(zip(examples, query_bundles)):
            query = _query_for_source(bundle, source)
            if not query:
                continue
            batch_indices.append(idx)
            task_ids.append(example.task_id)
            queries.append(query)

        if not queries:
            continue

        source_results = bm25_index.query(
            task_ids,
            queries,
            topk=_topk_for_source(args, source, base_topk),
        )
        for idx, candidates in zip(batch_indices, source_results):
            source_counts[idx][source] += len(candidates)
            for rank, candidate in enumerate(candidates):
                per_task_hits[idx].append((source, rank, candidate))

    merged_candidates = []
    trace_rows = []
    for example, hits, counts in zip(examples, per_task_hits, source_counts):
        candidates, trace = merge_candidates(
            hits,
            candidate_pool_size=getattr(args, "ucm_candidate_pool_size", 100),
            task_id=example.task_id,
            source_counts=counts,
        )
        merged_candidates.append(candidates)
        trace_rows.append(trace)

    return merged_candidates, trace_rows


def merge_candidates(hits, candidate_pool_size, task_id=None, source_counts=None):
    grouped = {}

    for source, rank, candidate in hits:
        key = candidate_key(candidate)
        if key not in grouped:
            grouped[key] = {
                "candidate": candidate,
                "sources": set(),
                "best_rank_by_source": {},
            }
        grouped[key]["sources"].add(source)
        best_rank_by_source = grouped[key]["best_rank_by_source"]
        best_rank_by_source[source] = min(rank, best_rank_by_source.get(source, rank))

    ranked = sorted(grouped.values(), key=_candidate_sort_key)
    capped = ranked[:candidate_pool_size]
    candidates = [_copy_with_multi_type(item["candidate"], item["sources"]) for item in capped]

    trace = {
        "task_id": task_id,
        "source_hits": dict(source_counts or Counter()),
        "raw_hits": len(hits),
        "merged_candidates": len(ranked),
        "candidate_pool_size": len(candidates),
    }
    return candidates, trace


def add_retrieval_trace_results(trace_rows, retrieved_codeblocks):
    for trace, candidates in zip(trace_rows, retrieved_codeblocks):
        trace["retrieved_sources"] = _retrieved_source_counts(candidates)
        trace["stop_rank"] = _find_stop_rank(candidates)
        trace.update(_retrieved_graph_rerank_stats(candidates))
        trace.update(_retrieved_typed_dependency_stats(candidates))
        trace.update(_retrieved_unified_graph_stats(candidates))
    return trace_rows


def write_retrieval_trace(output_dir, dataset_name, trace_rows, overwrite=False):
    if not output_dir or not dataset_name or not trace_rows:
        return

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    trace_path = os.path.join(dataset_dir, "ucm_retrieval_trace.jsonl")
    mode = "w" if overwrite else "a"
    with open(trace_path, mode, encoding="utf-8") as f_trace:
        for trace in trace_rows:
            f_trace.write(json.dumps(trace, ensure_ascii=False) + "\n")


def candidate_key(candidate):
    return (candidate.file_path, candidate.description, candidate.code_content)


def log_retrieval_trace(dataset_name, trace_rows, retriever_inputs=None):
    prefix = f"[UCM trace][{dataset_name}]" if dataset_name else "[UCM trace]"
    final_counts = [len(x) for x in retriever_inputs] if retriever_inputs is not None else None
    for idx, trace in enumerate(trace_rows):
        source_hits = ", ".join(
            f"{source}:{count}" for source, count in sorted(trace["source_hits"].items())
        )
        retriever_input = (
            final_counts[idx]
            if final_counts is not None
            else trace["candidate_pool_size"]
        )
        retrieved_sources = ", ".join(
            f"{source}:{count}" for source, count in sorted(trace.get("retrieved_sources", {}).items())
        )
        print(
            f"{prefix} task={trace['task_id']} "
            f"sources=({source_hits}) raw={trace['raw_hits']} "
            f"merged={trace['merged_candidates']} retriever_input={retriever_input} "
            f"retrieved=({retrieved_sources}) stop_rank={trace.get('stop_rank')}"
        )


def _ordered_sources(query_bundles):
    sources = []
    seen = set()
    for bundle in query_bundles:
        for view in bundle:
            if view.source not in seen:
                seen.add(view.source)
                sources.append(view.source)
    return sorted(sources, key=lambda source: SOURCE_PRIORITY.get(source, 99))


def _query_for_source(bundle, source):
    for view in bundle:
        if view.source == source:
            return view.query
    return ""


def _topk_for_source(args, source, base_topk):
    if source == "base":
        configured = getattr(args, "ucm_base_topk", 0)
        if configured > 0:
            return configured
        if base_topk is not None:
            return base_topk
    if source == "path":
        return getattr(args, "ucm_path_topk", 5)
    return getattr(args, "ucm_topk_per_path", 10)


def _candidate_sort_key(item):
    sources = item["sources"]
    ranks = item["best_rank_by_source"]
    best_rank = min(ranks.values())
    source_bonus = -len(sources)

    if "base" in ranks:
        return (0, ranks["base"], source_bonus, best_rank)
    if "draft" in ranks:
        return (1, ranks["draft"], source_bonus, best_rank)

    best_source_priority = min(SOURCE_PRIORITY.get(source, 99) for source in sources)
    return (2, best_source_priority, best_rank, source_bonus)


def _copy_with_multi_type(candidate, sources):
    best_source = min(sources, key=lambda source: SOURCE_PRIORITY.get(source, 99))
    copied = CodeBlock(
        candidate.file_path,
        candidate.description,
        candidate.code_content,
        candidate.language,
        f"multi_{best_source}",
    )
    copied._ucm_sources = tuple(sorted(sources, key=lambda source: SOURCE_PRIORITY.get(source, 99)))
    return copied


def _retrieved_source_counts(candidates):
    counts = Counter()
    for candidate in candidates:
        if _is_stop_block(candidate):
            break
        sources = getattr(candidate, "_ucm_sources", None)
        if sources:
            for source in sources:
                counts[source] += 1
        else:
            counts[getattr(candidate, "_type", "") or "unknown"] += 1
    return dict(counts)


def _find_stop_rank(candidates):
    for idx, candidate in enumerate(candidates, start=1):
        if _is_stop_block(candidate):
            return idx
    return None


def _retrieved_graph_rerank_stats(candidates):
    biases = []
    retriever_scores = []
    final_scores = []
    path_scores = []
    path_biases = []
    for candidate in candidates:
        if _is_stop_block(candidate):
            break
        bias = getattr(candidate, "_ucm_graph_rerank_bias", None)
        retriever_score = getattr(candidate, "_ucm_retriever_score", None)
        final_score = getattr(candidate, "_ucm_final_score", None)
        path_score = getattr(candidate, "_ucm_graph_path_score", None)
        path_bias = getattr(candidate, "_ucm_graph_path_bias", None)
        if bias is not None:
            biases.append(float(bias))
        if retriever_score is not None:
            retriever_scores.append(float(retriever_score))
        if final_score is not None:
            final_scores.append(float(final_score))
        if path_score is not None:
            path_scores.append(float(path_score))
        if path_bias is not None:
            path_biases.append(float(path_bias))

    if not biases:
        return {}

    nonzero_biases = [bias for bias in biases if abs(bias) > 1e-12]
    stats = {
        "graph_rerank_bias_count": len(biases),
        "graph_rerank_nonzero_bias_count": len(nonzero_biases),
        "graph_rerank_bias_avg": round(sum(biases) / len(biases), 6),
        "graph_rerank_bias_max": round(max(biases), 6),
    }
    if retriever_scores:
        stats["retriever_score_avg"] = round(sum(retriever_scores) / len(retriever_scores), 6)
    if final_scores:
        stats["final_score_avg"] = round(sum(final_scores) / len(final_scores), 6)
    if path_scores:
        stats["graph_path_score_avg"] = round(
            sum(path_scores) / len(path_scores), 6
        )
        stats["graph_path_score_max"] = round(max(path_scores), 6)
    if path_biases:
        stats["graph_path_bias_avg"] = round(
            sum(path_biases) / len(path_biases), 6
        )
    return stats


def _retrieved_typed_dependency_stats(candidates):
    relation_counts = Counter()
    origin_counts = Counter()
    matched_symbols = Counter()
    graph_only_count = 0

    for candidate in candidates:
        if _is_stop_block(candidate):
            break
        relation = getattr(candidate, "_ucm_graph_relation", None)
        if not relation or not relation.startswith("graph_typed_"):
            continue
        relation_counts[relation] += 1
        origin = getattr(candidate, "_ucm_graph_origin", None)
        if origin:
            origin_counts[origin] += 1
        matched_symbols.update(
            getattr(candidate, "_ucm_graph_matched_symbols", ())
        )
        sources = tuple(getattr(candidate, "_ucm_sources", ()))
        if sources and all(source.startswith("graph_") for source in sources):
            graph_only_count += 1

    if not relation_counts:
        return {}
    return {
        "typed_dependency_retrieved": sum(relation_counts.values()),
        "typed_dependency_graph_only_retrieved": graph_only_count,
        "typed_dependency_retrieved_relations": dict(relation_counts),
        "typed_dependency_retrieved_origins": dict(origin_counts),
        "typed_dependency_retrieved_symbols": dict(matched_symbols),
    }


def _retrieved_unified_graph_stats(candidates):
    relation_counts = Counter()
    origin_counts = Counter()
    confidences = []
    path_lengths = []
    evidence_candidates = 0
    multi_evidence_candidates = 0
    graph_only_count = 0

    for candidate in candidates:
        if _is_stop_block(candidate):
            break
        evidence = tuple(getattr(candidate, "_ucm_graph_evidence", ()))
        if not evidence:
            continue
        evidence_candidates += 1
        if len(evidence) > 1:
            multi_evidence_candidates += 1
        for item in evidence:
            relation_counts[item.get("relation", "unknown")] += 1
            origin_counts[item.get("origin", "unknown")] += 1
            confidences.append(float(item.get("confidence", 0.0)))
            path_lengths.append(int(item.get("path_length", 0)))
        sources = tuple(getattr(candidate, "_ucm_sources", ()))
        if sources and all(source.startswith("graph_") for source in sources):
            graph_only_count += 1

    if not evidence_candidates:
        return {}
    stats = {
        "unified_graph_retrieved_candidates": evidence_candidates,
        "unified_graph_retrieved_multi_evidence_candidates": multi_evidence_candidates,
        "unified_graph_retrieved_graph_only_candidates": graph_only_count,
        "unified_graph_retrieved_path_count": sum(relation_counts.values()),
        "unified_graph_retrieved_relations": dict(relation_counts),
        "unified_graph_retrieved_origins": dict(origin_counts),
    }
    if confidences:
        stats["unified_graph_retrieved_confidence_avg"] = round(
            sum(confidences) / len(confidences), 6
        )
        stats["unified_graph_retrieved_confidence_max"] = round(
            max(confidences), 6
        )
    if path_lengths:
        stats["unified_graph_retrieved_path_length_avg"] = round(
            sum(path_lengths) / len(path_lengths), 6
        )
    return stats


def _is_stop_block(candidate):
    return getattr(candidate, "file_path", None) == ""
