import torch.nn as nn
import torch
import math
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import CodeBlock
from utils.model_utils import resolve_model_path
from unified_context_graph import evidence_gate_passes


GRAPH_SOURCE_NAMES = {
    "graph_same_file",
    "graph_identifier",
    "graph_import_path",
    "graph_api_call",
    "graph_typed_call",
    "graph_typed_type",
    "graph_typed_def_use",
    "graph_unified",
}


def tokenize(text, tokenizer, max_length, is_query, extracted_import=''):
    """
    Converts text to a list of token ids.
    :param text: The text to be converted
    :param tokenizer: The tokenizer to use
    :param max_length: The maximum input length
    :param is_query: A flag indicating whether the text is a query
    :return: A list of token ids
    """
    if extracted_import:
        # import_tokens = tokenizer.tokenize(extracted_import)[-127:] + [tokenizer.sep_token]
        import_tokens = []
    else:
        import_tokens = []

    tokens = tokenizer.tokenize(text)
    if is_query:
        tokens = tokens[-(max_length - len(import_tokens)) + 4:]
    else:
        tokens = tokens[:(max_length - len(import_tokens)) - 4]
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + import_tokens + tokens + [tokenizer.sep_token]
    tokens_id = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = max_length - len(tokens_id)
    tokens_id += [tokenizer.pad_token_id] * padding_length

    return tokens_id


class CustomDataset(Dataset):
    """
    Custom dataset class for handling code blocks and queries.
    :param max_length: The maximum input length
    :param tokenizer: The tokenizer used
    :param examples: The samples in the dataset
    :param is_query: A flag indicating whether it is a query
    """
    def __init__(self, max_length, tokenizer, examples, query=False, extracted_imports=None):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.examples = examples
        self.query = query
        self.extracted_imports = extracted_imports

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = str(self.examples[idx])
        extracted_import = str(self.extracted_imports[idx]) if self.extracted_imports else ''
        tokens_id = tokenize(text, self.tokenizer, self.max_length, self.query, extracted_import)
        return torch.tensor(tokens_id, dtype=torch.long)



class Retriever(nn.Module):
    """
    Retriever model, used to compute sentence embeddings and retrieve similar code blocks.
    :param args: A namespace containing configuration parameters
    """
    def __init__(self, args):
        super(Retriever, self).__init__()
        retriever_model_path = resolve_model_path(args.retriever_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_model_path, local_files_only=True)
        self.args = args
        if self.args.disable_retriever is False:
            self.model = AutoModel.from_pretrained(retriever_model_path, local_files_only=True)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()

    def forward(self, source_ids):
        """
        Forward propagation function, used to generate the embedding representation of the input.
        :param input_ids: The sequence of input IDs
        :return: The embedding representation
        """
        mask = source_ids.ne(self.tokenizer.pad_token_id)
        token_embeddings = self.model(source_ids, attention_mask=mask)[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def retrieve(self, queries, candidate_codeblocks, topk, extracted_imports=None):
        """
        Retrieval function, used to retrieve the most relevant code blocks from a list of candidate code blocks for each query.
        :param queries: A list of queries
        :param candidate_codeblocks: A list of candidate code blocks
        :param topk: The number of top-k code blocks to return for each query
        :return: A list of top-k code blocks for each query
        """
        query_dataset = CustomDataset(self.args.retriever_query_context_length, self.tokenizer, queries, query=True, extracted_imports=extracted_imports)
        query_dataloader = DataLoader(query_dataset, batch_size=self.args.retriever_batch_size, shuffle=False, num_workers=self.args.num_workers)
        candidate_numbers = [len(x) for x in candidate_codeblocks]
        candidate_codeblocks = [x for y in candidate_codeblocks for x in y]
        code_dataset = CustomDataset(self.args.retriever_candidate_context_length, self.tokenizer, candidate_codeblocks, query=False)
        code_dataloader = DataLoader(code_dataset, batch_size=self.args.retriever_batch_size, shuffle=False, num_workers=self.args.num_workers)
        code_dataloader = tqdm(code_dataloader, desc="Encoding Code Blocks") if self.args.enable_tqdm else code_dataloader
        query_embeddings = []
        code_embeddings = []
        with torch.no_grad():
            for batch in query_dataloader:
                batch = batch.cuda()
                query_embeddings.append(self.forward(batch))
            for batch in code_dataloader:
                batch = batch.cuda()
                code_embeddings.append(self.forward(batch))
        query_embeddings = torch.cat(query_embeddings, dim=0)
        code_embeddings = torch.cat(code_embeddings, dim=0)

        scores = torch.mm(query_embeddings, code_embeddings.t())
        scores = scores.cpu().numpy()

        topk_codeblocks = []  # Stores top-k codeblocks for each query
        start_idx = 0
        for i, num_candidates in enumerate(candidate_numbers):
            if num_candidates == 0:
                topk_codeblocks.append([])  # If there are no candidates for this query, add an empty list
                continue
            query_scores = scores[i][start_idx:start_idx+num_candidates]  # Get scores for the current query
            query_candidates = candidate_codeblocks[start_idx:start_idx+num_candidates]
            final_scores, graph_biases = _apply_graph_rerank_scores(self.args, query_scores, query_candidates)
            topk_indices_query = _select_topk_indices(
                self.args,
                final_scores,
                query_candidates,
                topk,
                self.tokenizer,
            )
            topk_codeblocks_query = []
            for idx in topk_indices_query:
                codeblock = candidate_codeblocks[start_idx + idx]
                codeblock._ucm_retriever_score = float(query_scores[idx])
                codeblock._ucm_final_score = float(final_scores[idx])
                codeblock._ucm_graph_rerank_bias = float(graph_biases[idx])
                topk_codeblocks_query.append(codeblock)

            if len(topk_codeblocks_query) < topk:
                topk_codeblocks_query += [CodeBlock("","Don't need cross file context to completion", "", topk_codeblocks_query[0].language, '')] * (topk - len(topk_codeblocks_query))
            topk_codeblocks.append(topk_codeblocks_query)
            start_idx += num_candidates
        return topk_codeblocks


def _apply_graph_rerank_scores(args, query_scores, query_candidates):
    if not getattr(args, "ucm_graph_enable_rerank", False):
        return query_scores, query_scores * 0.0

    graph_biases = query_scores * 0.0
    alpha = getattr(args, "ucm_graph_rerank_alpha", 0.03)
    distance_penalty = getattr(args, "ucm_graph_distance_penalty", 0.005)

    if getattr(args, "ucm_graph_enable_path_rerank", False):
        score_scale = max(
            1e-6, getattr(args, "ucm_graph_path_score_scale", 4.0)
        )
        multi_bonus = max(
            0.0, getattr(args, "ucm_graph_path_multi_relation_bonus", 0.004)
        )
        query_bonus = max(
            0.0, getattr(args, "ucm_graph_path_query_origin_bonus", 0.003)
        )
        for idx, candidate in enumerate(query_candidates):
            evidence = tuple(getattr(candidate, "_ucm_graph_evidence", ()))
            graph_score = float(getattr(candidate, "_ucm_graph_score", 0.0) or 0.0)
            relations = {
                item.get("relation") for item in evidence if item.get("relation")
            }
            origins = {item.get("origin") for item in evidence if item.get("origin")}
            graph_bias = alpha * math.tanh(graph_score / score_scale)
            graph_bias += multi_bonus * min(3, max(0, len(relations) - 1))
            if "query" in origins:
                graph_bias += query_bonus
            graph_bias += _graph_source_prior(args, candidate)
            edge_distance = getattr(candidate, "_ucm_graph_edge_distance", None)
            if edge_distance is not None:
                graph_bias -= distance_penalty * max(0.0, float(edge_distance))
            candidate._ucm_graph_path_score = graph_score
            candidate._ucm_graph_evidence_count = len(evidence)
            candidate._ucm_graph_relation_diversity = len(relations)
            candidate._ucm_graph_path_bias = graph_bias
            graph_biases[idx] = graph_bias
        return query_scores + graph_biases, graph_biases

    graph_norm_scores = _normalized_graph_scores(query_candidates)

    for idx, candidate in enumerate(query_candidates):
        graph_bias = alpha * graph_norm_scores[idx]
        graph_bias += _graph_source_prior(args, candidate)
        edge_distance = getattr(candidate, "_ucm_graph_edge_distance", None)
        if edge_distance is not None:
            graph_bias -= distance_penalty * max(0.0, float(edge_distance))
        graph_biases[idx] = graph_bias

    return query_scores + graph_biases, graph_biases


def _normalized_graph_scores(query_candidates):
    graph_scores = []
    for candidate in query_candidates:
        score = getattr(candidate, "_ucm_graph_score", None)
        graph_scores.append(float(score) if score is not None else None)

    valid_scores = [score for score in graph_scores if score is not None]
    if not valid_scores:
        return [0.0 for _ in query_candidates]

    score_min = min(valid_scores)
    score_max = max(valid_scores)
    if score_max <= score_min:
        return [1.0 if score is not None else 0.0 for score in graph_scores]

    return [
        (score - score_min) / (score_max - score_min) if score is not None else 0.0
        for score in graph_scores
    ]


def _graph_source_prior(args, candidate):
    priors = []
    for source in _candidate_graph_sources(candidate):
        if source == "graph_same_file":
            priors.append(getattr(args, "ucm_graph_source_prior_same_file", 0.02))
        elif source == "graph_identifier":
            priors.append(getattr(args, "ucm_graph_source_prior_identifier", 0.015))
        elif source == "graph_import_path":
            priors.append(getattr(args, "ucm_graph_source_prior_import", 0.005))
        elif source == "graph_api_call":
            priors.append(getattr(args, "ucm_graph_source_prior_api_call", 0.025))
    return max(priors) if priors else 0.0


def _candidate_graph_sources(candidate):
    sources = getattr(candidate, "_ucm_sources", None)
    if not sources:
        sources = (getattr(candidate, "_type", ""),)
    return [source for source in sources if source in GRAPH_SOURCE_NAMES]


def _select_topk_indices(args, final_scores, candidates, topk, tokenizer=None):
    ranked_indices = final_scores.argsort()[::-1]
    max_graph_blocks = max(0, getattr(args, "ucm_graph_max_selected", 0))
    max_graph_tokens = max(
        0, getattr(args, "ucm_graph_max_selected_tokens", 0)
    )
    if max_graph_blocks == 0 and max_graph_tokens == 0:
        return ranked_indices[:topk]

    selected = []
    selected_graph_blocks = 0
    selected_graph_tokens = 0
    for idx in ranked_indices:
        candidate = candidates[idx]
        if _is_graph_only_candidate(candidate):
            if getattr(args, "ucm_graph_enable_evidence_gate", False):
                evidence = tuple(getattr(candidate, "_ucm_graph_evidence", ()))
                min_paths = max(
                    1, getattr(args, "ucm_graph_new_candidate_min_paths", 2)
                )
                min_confidence = max(
                    0.0,
                    getattr(
                        args,
                        "ucm_graph_new_candidate_min_confidence",
                        3.0,
                    ),
                )
                if not evidence_gate_passes(
                    evidence,
                    min_paths=min_paths,
                    min_confidence=min_confidence,
                ):
                    continue
            if max_graph_blocks and selected_graph_blocks >= max_graph_blocks:
                continue
            candidate_tokens = 0
            if max_graph_tokens and tokenizer is not None:
                candidate_tokens = len(
                    tokenizer.encode(str(candidate), add_special_tokens=False)
                )
                if selected_graph_tokens + candidate_tokens > max_graph_tokens:
                    continue
            selected_graph_blocks += 1
            selected_graph_tokens += candidate_tokens

        selected.append(idx)
        if len(selected) >= topk:
            break
    return selected


def _is_graph_only_candidate(candidate):
    sources = getattr(candidate, "_ucm_sources", None)
    if not sources:
        sources = (getattr(candidate, "_type", ""),)
    nonempty_sources = [source for source in sources if source]
    return bool(nonempty_sources) and all(
        source in GRAPH_SOURCE_NAMES for source in nonempty_sources
    )

