import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from utils.eval_utils import is_identifier
from utils.model_utils import resolve_model_path
import re

class CustomDataset(Dataset):
    """
    A dataset class for code generation.

    Args:
        args: Configuration parameters.
        tokenizer: Tokenizer.
        examples: A collection of examples.
        retrieved_codeblocks: Retrieved code blocks.
    """
    def __init__(self, args, tokenizer, examples, retrieved_codeblocks, generation=False):
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        self.retrieved_codeblocks = retrieved_codeblocks
        self.generation = generation

    def __len__(self):
        return len(self.examples)

    def construct_prompts(self,example,retrieved_codeblocks):
        if getattr(self.args, "inference_type", None) == "zeroshot":
            # Pure generator-only completion: pass the visible code prefix as-is.
            # Token truncation in __getitem__ keeps the most recent 4K (or the
            # configured context window) without adding paths or RAG evidence.
            return example.left_context

        filter_codeblocks = []
        for x in retrieved_codeblocks:
            # Keep only the blocks before the empty block
            if x.file_path != "":
                filter_codeblocks.append(x)
            else:
                break
        crossfile_context = "\n\n".join(
            [self._render_codeblock(block) for block in filter_codeblocks]
        )
        crossfile_context = self.tokenizer.encode(crossfile_context[:self.args.generator_max_crossfile_length*10], add_special_tokens=False)[:self.args.generator_max_crossfile_length]
        path_context = f"\n\n# file path: {example.file_path}\n\n"
        path_context = self.tokenizer.encode(path_context, add_special_tokens=False)
        allowed_prompt_length = self.args.generator_max_context_length - (len(crossfile_context)+len(path_context)+10)
        if allowed_prompt_length > 0:
            infile_context = self.tokenizer.encode(example.left_context, add_special_tokens=False)[-allowed_prompt_length:]
        else:
            infile_context = []

        prompt = self.tokenizer.decode(crossfile_context + path_context + infile_context)
        return prompt

    def _render_codeblock(self, block):
        block_text = str(block)
        if getattr(self.args, "ucm_graph_show_high_confidence_paths", False):
            evidence = tuple(getattr(block, "_ucm_graph_evidence", ()))
            sources = tuple(getattr(block, "_ucm_sources", ()))
            graph_only = bool(sources) and all(
                source.startswith("graph_") for source in sources
            )
            min_confidence = getattr(
                self.args, "ucm_graph_prompt_min_confidence", 3.0
            )
            eligible = [
                item
                for item in evidence
                if float(item.get("confidence", 0.0)) >= min_confidence
            ]
            if graph_only and eligible:
                dominant = eligible[0]
                path = " -> ".join(dominant.get("path", ()))
                comment_label = "#" if block.language == "python" else "//"
                return f"{comment_label} context path: {path}\n{block_text}"

        if not getattr(self.args, "ucm_graph_show_relations_in_prompt", False):
            return block_text

        relation = getattr(block, "_ucm_graph_relation", None)
        symbols = getattr(block, "_ucm_graph_matched_symbols", ())
        if not relation or not symbols:
            return block_text

        max_symbols = max(
            1, getattr(self.args, "ucm_graph_relation_max_symbols", 3)
        )
        relation_label = {
            "graph_typed_call": "calls",
            "graph_typed_type": "uses type",
            "graph_typed_def_use": "uses definition",
        }.get(relation, relation)
        origin = getattr(block, "_ucm_graph_origin", None) or "seed"
        comment_label = "#" if block.language == "python" else "//"
        symbol_text = ", ".join(symbols[:max_symbols])
        dependency_header = (
            f"{comment_label} dependency: {origin} --{relation_label}--> "
            f"{symbol_text}"
        )
        return f"{dependency_header}\n{block_text}"

    def __getitem__(self, idx):
        example = self.examples[idx]
        retrieved_codeblocks = self.retrieved_codeblocks[idx]
        prompt = self.construct_prompts(example,retrieved_codeblocks)

        prompt_ids = self.tokenizer.encode(prompt)[-self.args.generator_max_context_length:]
        if self.generation:
             padding_length = self.args.generator_max_context_length - len(prompt_ids)
             input_ids = [self.tokenizer.pad_token_id] * padding_length + prompt_ids
             return torch.tensor(input_ids)

        target_ids = self.tokenizer.encode(example.target_code, add_special_tokens=False)[:self.args.generator_max_generation_length]

        input_ids = prompt_ids + target_ids
        labels = [-100 for _ in prompt_ids] + target_ids

        padding_length = self.args.generator_max_context_length + self.args.generator_max_generation_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        labels = [-100] * padding_length + labels

        return torch.tensor(input_ids), torch.tensor(labels)


class Model(nn.Module):
    def __init__(self, generator_model_path, tokenizer, max_generation_length=64):
        super(Model, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            resolve_model_path(generator_model_path),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length

    def forward(self, inputs=None, labels=None, lang='python', weighted_keywords=False):
        """
        Forward propagation method for calculating loss.
        :param inputs: Input data.
        :param labels: Label data.
        :return: The average loss per sample.
        """
        if labels is not None:
            logits = self.base_model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id))[0]
            logits = logits[:, :-1]
            labels = labels[:, 1:]

            label_tokens = [self.tokenizer.convert_ids_to_tokens(id.item()) if id != -100 else '<pad>' for id in labels.reshape(-1)]

            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction='none')

            if weighted_keywords:
                id_weight = 3
                first_token_weight = 5

                weights = torch.tensor([
                    first_token_weight if i < 1 else id_weight if is_identifier(token, lang) and any(c.isalpha() or c.isdigit() or c == '_' for c in token) else 1
                    for i, token in enumerate(label_tokens)
                ], dtype=torch.float).cuda()

                loss = loss * weights

            loss_per_label = loss.reshape(labels.size(0), -1).sum(dim=1) / labels.ne(-100).sum(dim=1)

            return loss_per_label
        else:
            generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1)+self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id)
            return generated_ids[:, inputs.size(1):]



class Generator:
    """
    Code generator class.

    Args:
        args: Configuration parameters.
    """
    def __init__(self, args):
        generator_model_path = resolve_model_path(args.generator_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_path, local_files_only=True)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not args.disable_generator:
            self.model = Model(generator_model_path, self.tokenizer)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()

        self.args = args

    def evaluate(self, examples, retrieved_codeblocks):
        """
        Evaluates the generated code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.

        Returns:
            A list of loss values.
        """
        losses = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm)
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = [x.cuda() for x in batch]
                loss_per_label = self.model(inputs, labels, lang=examples[0].language, weighted_keywords=self.args.weighted_keywords)
                losses.extend(loss_per_label.tolist())
                current_ppl = np.exp(np.mean(losses))
                pbar.set_description(f"Loss/PPL: {np.mean(losses):.3f}/{current_ppl:.3f}")

        return losses

    def generate(self, examples, retrieved_codeblocks, max_generation_length):
        """
        Generates code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.
            max_generation_length: Maximum length of generation.

        Returns:
            A list of generated codes.
        """
        generated_codes = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks,generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
        if hasattr(self.model, "module"):
            self.model.module.max_generation_length = max_generation_length
        else:
            self.model.max_generation_length = max_generation_length

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
        with torch.no_grad():
            for batch in pbar:
                generated_codes.append(self.model(batch.cuda()))
        generated_codes = torch.cat(generated_codes,0)
        return  [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_codes]


