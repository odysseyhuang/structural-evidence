def apply_candidate_gate(candidate_codeblocks, args):
    if not getattr(args, "enable_context_gate", False):
        return candidate_codeblocks

    return [_gate_candidate_list(candidates, args) for candidates in candidate_codeblocks]


def apply_retrieved_context_gate(retrieved_codeblocks, args):
    if not getattr(args, "enable_context_gate", False):
        return retrieved_codeblocks

    threshold = getattr(args, "ucm_gate_stop_rank_threshold", 2)
    gated = []
    for candidates in retrieved_codeblocks:
        stop_rank = find_stop_rank(candidates)
        if stop_rank is not None and stop_rank <= threshold:
            gated.append(candidates[:stop_rank])
        else:
            gated.append(candidates)
    return gated


def find_stop_rank(candidates):
    for idx, candidate in enumerate(candidates, start=1):
        if is_stop_block(candidate):
            return idx
    return None


def is_stop_block(candidate):
    return getattr(candidate, "file_path", None) == ""


def _gate_candidate_list(candidates, args):
    max_auxiliary = getattr(args, "ucm_gate_max_auxiliary_blocks", 2)
    allow_path_only = getattr(args, "ucm_gate_allow_path_only", 0)

    selected = []
    auxiliary_count = 0
    path_only_count = 0

    for candidate in candidates:
        sources = _candidate_sources(candidate)
        if "base" in sources:
            selected.append(candidate)
            continue

        if sources == {"path"}:
            if path_only_count < allow_path_only:
                selected.append(candidate)
                path_only_count += 1
            continue

        if auxiliary_count < max_auxiliary:
            selected.append(candidate)
            auxiliary_count += 1

    if not selected and candidates:
        selected.append(candidates[0])

    return selected


def _candidate_sources(candidate):
    sources = getattr(candidate, "_ucm_sources", None)
    if sources:
        return set(sources)

    candidate_type = getattr(candidate, "_type", "")
    if candidate_type.startswith("multi_"):
        return {candidate_type[len("multi_"):]}
    return {candidate_type or "unknown"}
