import ast
import math
import re
from collections import defaultdict

from typed_dependency_graph import block_key, extract_dependency_facts


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
QUALIFIED_CALL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
PYTHON_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import\s+([^\n]+)|"
    r"import\s+([^\n]+))"
)
PYTHON_CLASS_RE = re.compile(
    r"(?m)^\s*class\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*:"
)
PYTHON_FUNCTION_RE = re.compile(
    r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*"
    r"\(([^)]*)\)\s*(?:->\s*([^:\n]+))?\s*:"
)
PYTHON_ANNOTATED_BINDING_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)"
)
PYTHON_CONSTRUCTOR_BINDING_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s*\("
)
JAVA_IMPORT_RE = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([\w.*]+)\s*;")
JAVA_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([\w.]+)\s*;")
JAVA_TYPE_RE = re.compile(
    r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)"
    r"(?:\s+extends\s+([A-Za-z_$][\w$., <>]*))?"
    r"(?:\s+implements\s+([A-Za-z_$][\w$., <>]*))?"
)
JAVA_METHOD_RE = re.compile(
    r"(?m)^\s*(?:(?:public|protected|private|static|final|abstract|"
    r"synchronized|native|default|strictfp)\s+)*"
    r"(?:<[^>{};]+>\s+)?([A-Za-z_$][\w$<>,.?\[\]]*)\s+"
    r"([A-Za-z_$][\w$]*)\s*\(([^;{}]*)\)\s*"
    r"(?:throws\s+[^\{]+)?(?:\{|$)"
)
JAVA_BINDING_RE = re.compile(
    r"\b(?:final\s+)?([A-Za-z_$][\w$<>,.?\[\]]*)\s+"
    r"([A-Za-z_$][\w$]*)\s*(?==|;|,)"
)
JAVA_CONTROL_RE = re.compile(
    r"\b(if|else\s+if|for|while|switch|catch|try|synchronized)\s*"
    r"(?:\(([^)]*)\))?"
)
PYTHON_CONTROL_RE = re.compile(
    r"^\s*(if|elif|for|while|try|except|with)\b([^:]*)"
)

BUILTIN_TYPES = {
    "Any", "None", "Object", "String", "bool", "boolean", "byte", "char",
    "dict", "double", "float", "int", "integer", "list", "long", "map",
    "object", "set", "short", "str", "string", "tuple", "void",
}

DEFAULT_RELATION_WEIGHTS = {
    "unified_receiver_member": 3.2,
    "unified_override": 2.9,
    "unified_member_of": 2.6,
    "unified_import_resolution": 2.5,
    "unified_inheritance": 2.3,
    "unified_type_definition": 2.0,
    "unified_call_definition": 1.8,
    "unified_signature_type": 1.7,
    "unified_def_use": 1.3,
    "unified_control_dependency": 0.8,
}


class UnifiedContextGraphIndex:
    """Repository semantic graph connected to a visible-prefix query graph."""

    def __init__(self, code_blocks_by_task):
        self.code_blocks_by_task = code_blocks_by_task
        self.task_indices = {}

    def neighbors(
        self,
        task_id,
        source_text,
        language,
        max_results,
        max_df,
        origin="query",
        exclude_block_key=None,
        exclude_file_path="",
        weights=None,
        max_evidence_per_candidate=12,
    ):
        if max_results <= 0:
            return []

        index = self._ensure_task(task_id)
        facts = extract_unified_facts(
            source_text,
            language,
            visible_prefix=(origin == "query"),
        )
        relation_weights = dict(DEFAULT_RELATION_WEIGHTS)
        relation_weights.update(weights or {})
        proposals = {}

        def add(idx, relation, symbols, path, ambiguity=1, path_length=None):
            block = index["blocks"][idx]
            if exclude_block_key is not None and block_key(block) == exclude_block_key:
                return
            if exclude_file_path and _same_path(block.file_path, exclude_file_path):
                return
            ambiguity = max(1, int(ambiguity))
            path_length = path_length or max(1, len(path) - 1)
            confidence = float(relation_weights.get(relation, 1.0))
            confidence += 1.0 / ambiguity
            confidence -= 0.12 * max(0, path_length - 2)
            evidence = {
                "relation": relation,
                "origin": origin,
                "symbols": tuple(sorted(set(symbols))),
                "path": tuple(path),
                "confidence": round(max(0.0, confidence), 6),
                "path_length": path_length,
                "ambiguity": ambiguity,
            }
            proposal = proposals.setdefault(idx, {"block": block, "evidence": []})
            proposal["evidence"] = merge_evidence(
                proposal["evidence"],
                [evidence],
                max_items=max_evidence_per_candidate,
            )

        def add_definition_matches(symbols, required_kinds, relation, path_label):
            for symbol in sorted(symbols):
                entries = index["definitions"].get(symbol, ())
                if not entries or len(entries) > max_df:
                    continue
                for idx, kinds in entries:
                    if required_kinds and not required_kinds.intersection(kinds):
                        continue
                    add(
                        idx,
                        relation,
                        (symbol,),
                        (origin, f"{path_label}:{symbol}", f"declares:{symbol}", "block"),
                        ambiguity=len(entries),
                        path_length=2,
                    )

        add_definition_matches(
            facts["type_refs"], {"type"}, "unified_type_definition", "uses_type"
        )
        add_definition_matches(
            facts["calls"], {"callable"}, "unified_call_definition", "calls"
        )
        add_definition_matches(
            facts["uses"], {"type", "callable"}, "unified_def_use", "uses"
        )

        expected_types = facts["expected_types"] - BUILTIN_TYPES
        add_definition_matches(
            expected_types,
            {"type"},
            "unified_signature_type",
            "expects_type",
        )

        for receiver, member in sorted(facts["receiver_calls"]):
            receiver_types = set(facts["variable_types"].get(receiver, ()))
            if receiver in {"self", "this"}:
                receiver_types.update(facts["owners"])
            entries = index["members"].get(member, ())
            if not entries or len(entries) > max_df:
                continue
            for idx, owner in entries:
                if receiver_types and owner in receiver_types:
                    add(
                        idx,
                        "unified_receiver_member",
                        (owner, member),
                        (
                            origin,
                            f"reaching_type:{receiver}->{owner}",
                            f"calls_member:{member}",
                            f"member_of:{owner}",
                            "block",
                        ),
                        ambiguity=len(entries),
                        path_length=3,
                    )
                elif receiver_types and any(
                    _is_related_owner(index, receiver_type, owner)
                    for receiver_type in receiver_types
                ):
                    add(
                        idx,
                        "unified_override",
                        tuple(sorted(receiver_types | {owner, member})),
                        (
                            origin,
                            f"receiver:{receiver}",
                            f"calls_member:{member}",
                            f"inheritance_owner:{owner}",
                            "block",
                        ),
                        ambiguity=len(entries),
                        path_length=3,
                    )

        for member in sorted(facts["calls"]):
            entries = index["members"].get(member, ())
            if not entries or len(entries) > max_df:
                continue
            for idx, owner in entries:
                if owner and owner in facts["type_refs"]:
                    add(
                        idx,
                        "unified_member_of",
                        (owner, member),
                        (
                            origin,
                            f"uses_type:{owner}",
                            f"calls:{member}",
                            f"member_of:{owner}",
                            "block",
                        ),
                        ambiguity=len(entries),
                        path_length=3,
                    )

        import_tokens = facts["imports"]
        for token in sorted(import_tokens):
            candidates = set(index["path_tokens"].get(token, ()))
            candidates.update(idx for idx, _ in index["definitions"].get(token, ()))
            if not candidates or len(candidates) > max_df:
                continue
            for idx in candidates:
                add(
                    idx,
                    "unified_import_resolution",
                    (token,),
                    (origin, f"imports:{token}", f"resolves_to:{token}", "block"),
                    ambiguity=len(candidates),
                    path_length=2,
                )

        for type_name in sorted(facts["type_refs"] | expected_types):
            related_owners = set(index["bases_by_owner"].get(type_name, ()))
            related_owners.update(index["derived_by_base"].get(type_name, ()))
            for owner in sorted(related_owners):
                entries = index["owner_blocks"].get(owner, ())
                if not entries or len(entries) > max_df:
                    continue
                for idx in entries:
                    add(
                        idx,
                        "unified_inheritance",
                        (type_name, owner),
                        (
                            origin,
                            f"uses_type:{type_name}",
                            f"inheritance:{type_name}<->{owner}",
                            f"declares_type:{owner}",
                            "block",
                        ),
                        ambiguity=len(entries),
                        path_length=3,
                    )

        if facts["control_symbols"]:
            for proposal in proposals.values():
                matched = set()
                for evidence in proposal["evidence"]:
                    matched.update(evidence["symbols"])
                guarded = matched.intersection(facts["control_symbols"])
                if not guarded:
                    continue
                control_evidence = {
                    "relation": "unified_control_dependency",
                    "origin": origin,
                    "symbols": tuple(sorted(guarded)),
                    "path": (
                        origin,
                        "visible_control",
                        "guard_symbol:" + ",".join(sorted(guarded)),
                        "block",
                    ),
                    "confidence": round(
                        relation_weights["unified_control_dependency"] + 0.5, 6
                    ),
                    "path_length": 2,
                    "ambiguity": 1,
                }
                proposal["evidence"] = merge_evidence(
                    proposal["evidence"],
                    [control_evidence],
                    max_items=max_evidence_per_candidate,
                )

        ranked = []
        for proposal in proposals.values():
            evidence = proposal["evidence"]
            proposal["score"] = aggregate_evidence_score(evidence)
            proposal["relations"] = tuple(
                sorted({item["relation"] for item in evidence})
            )
            proposal["symbols"] = tuple(
                sorted({symbol for item in evidence for symbol in item["symbols"]})
            )
            ranked.append(proposal)
        ranked.sort(
            key=lambda item: (
                -item["score"],
                -len(item["relations"]),
                block_key(item["block"]),
            )
        )
        return ranked[:max_results], summarize_query_facts(facts)

    def _ensure_task(self, task_id):
        if task_id in self.task_indices:
            return self.task_indices[task_id]

        blocks = self.code_blocks_by_task.get(task_id, [])
        facts_by_block = []
        definitions = defaultdict(list)
        members = defaultdict(list)
        owner_blocks = defaultdict(list)
        bases_by_owner = defaultdict(set)
        derived_by_base = defaultdict(set)
        path_tokens = defaultdict(list)

        for idx, block in enumerate(blocks):
            facts = extract_unified_facts(
                getattr(block, "code_content", ""),
                getattr(block, "language", ""),
                visible_prefix=False,
            )
            facts_by_block.append(facts)
            for symbol, kinds in facts["definitions"].items():
                definitions[symbol].append((idx, frozenset(kinds)))
            for member, owners in facts["members"].items():
                if owners:
                    for owner in owners:
                        members[member].append((idx, owner))
                else:
                    members[member].append((idx, ""))
            for owner in facts["owners"]:
                owner_blocks[owner].append(idx)
            for owner, bases in facts["bases"].items():
                bases_by_owner[owner].update(bases)
                for base in bases:
                    derived_by_base[base].add(owner)
            for token in _path_tokens(getattr(block, "file_path", "")):
                path_tokens[token].append(idx)

        task_index = {
            "blocks": blocks,
            "facts": facts_by_block,
            "definitions": dict(definitions),
            "members": dict(members),
            "owner_blocks": dict(owner_blocks),
            "bases_by_owner": {k: frozenset(v) for k, v in bases_by_owner.items()},
            "derived_by_base": {k: frozenset(v) for k, v in derived_by_base.items()},
            "path_tokens": dict(path_tokens),
        }
        self.task_indices[task_id] = task_index
        return task_index


def extract_unified_facts(text, language, visible_prefix=False):
    text = text or ""
    if (language or "").lower() == "python":
        structural = _extract_python_structure(text)
    else:
        structural = _extract_java_structure(text)

    dependency = extract_dependency_facts(text, language)
    structural["definitions"] = dependency["definitions"]
    structural["calls"].update(dependency["calls"])
    structural["type_refs"].update(dependency["types"])
    structural["uses"].update(dependency["uses"])
    structural["type_refs"].difference_update(BUILTIN_TYPES)
    if visible_prefix:
        controls = _extract_visible_controls(text, language)
        structural["control_symbols"].update(controls["symbols"])
        structural["control_kinds"].update(controls["kinds"])
    return structural


def aggregate_evidence_score(evidence):
    if not evidence:
        return 0.0
    confidences = sorted(
        (max(0.0, float(item.get("confidence", 0.0))) for item in evidence),
        reverse=True,
    )
    relations = {item.get("relation") for item in evidence if item.get("relation")}
    origins = {item.get("origin") for item in evidence if item.get("origin")}
    score = confidences[0] + 0.25 * sum(confidences[1:4])
    score += 0.2 * max(0, len(relations) - 1)
    if "query" in origins and len(relations) > 1:
        score += 0.25
    if {
        "unified_type_definition",
        "unified_call_definition",
    }.issubset(relations) or "unified_receiver_member" in relations:
        score += 0.35
    return round(score, 6)


def merge_evidence(existing, incoming, max_items=12):
    merged = {}
    for item in list(existing or ()) + list(incoming or ()):
        normalized = dict(item)
        normalized["symbols"] = tuple(normalized.get("symbols", ()))
        normalized["path"] = tuple(normalized.get("path", ()))
        signature = (
            normalized.get("relation"),
            normalized.get("origin"),
            normalized["symbols"],
            normalized["path"],
        )
        previous = merged.get(signature)
        if previous is None or normalized.get("confidence", 0.0) > previous.get(
            "confidence", 0.0
        ):
            merged[signature] = normalized
    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            item.get("path_length", 99),
            item.get("relation", ""),
            item.get("path", ()),
        ),
    )
    return ranked[:max(1, int(max_items))]


def evidence_gate_passes(evidence, min_paths=2, min_confidence=3.0):
    evidence = tuple(evidence or ())
    max_confidence = max(
        (float(item.get("confidence", 0.0)) for item in evidence),
        default=0.0,
    )
    return len(evidence) >= max(1, int(min_paths)) or max_confidence >= max(
        0.0, float(min_confidence)
    )


def summarize_query_facts(facts):
    return {
        "calls": len(facts["calls"]),
        "type_refs": len(facts["type_refs"]),
        "uses": len(facts["uses"]),
        "imports": len(facts["imports"]),
        "receiver_calls": len(facts["receiver_calls"]),
        "variable_type_bindings": sum(len(v) for v in facts["variable_types"].values()),
        "control_symbols": len(facts["control_symbols"]),
        "control_kinds": tuple(sorted(facts["control_kinds"])),
        "expected_types": tuple(sorted(facts["expected_types"])),
    }


def _empty_structure():
    return {
        "definitions": {},
        "owners": set(),
        "members": defaultdict(set),
        "bases": defaultdict(set),
        "imports": set(),
        "variable_types": defaultdict(set),
        "calls": set(),
        "receiver_calls": set(),
        "type_refs": set(),
        "uses": set(),
        "return_types": set(),
        "parameter_types": set(),
        "expected_types": set(),
        "control_symbols": set(),
        "control_kinds": set(),
    }


class _PythonStructureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.facts = _empty_structure()
        self.owner_stack = []

    def visit_ClassDef(self, node):
        self.facts["owners"].add(node.name)
        self.facts["bases"][node.name].update(_annotation_names_many(node.bases))
        self.owner_stack.append(node.name)
        self.generic_visit(node)
        self.owner_stack.pop()

    def visit_FunctionDef(self, node):
        if self.owner_stack:
            self.facts["members"][node.name].add(self.owner_stack[-1])
        return_types = _annotation_names(node.returns)
        self.facts["return_types"].update(return_types)
        self.facts["expected_types"].update(return_types)
        for argument in list(node.args.args) + list(node.args.kwonlyargs):
            names = _annotation_names(argument.annotation)
            self.facts["parameter_types"].update(names)
            self.facts["variable_types"][argument.arg].update(names)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.facts["variable_types"][node.target.id].update(
                _annotation_names(node.annotation)
            )
        self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            constructor = _call_name(node.value.func)
            if constructor and constructor[:1].isupper():
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.facts["variable_types"][target.id].add(constructor)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            receiver = _expression_tail(node.func.value)
            if receiver:
                self.facts["receiver_calls"].add((receiver, node.func.attr))
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.facts["imports"].update(_qualified_tokens(alias.name))

    def visit_ImportFrom(self, node):
        self.facts["imports"].update(_qualified_tokens(node.module or ""))
        for alias in node.names:
            self.facts["imports"].add(alias.name)


def _extract_python_structure(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _extract_python_structure_regex(text)
    visitor = _PythonStructureVisitor()
    visitor.visit(tree)
    return visitor.facts


def _extract_python_structure_regex(text):
    facts = _empty_structure()
    owners = [match[0] for match in PYTHON_CLASS_RE.findall(text)]
    facts["owners"].update(owners)
    for owner, bases_text in PYTHON_CLASS_RE.findall(text):
        facts["bases"][owner].update(_type_names(bases_text))
    current_owner = owners[-1] if owners else ""
    for function, params, returns in PYTHON_FUNCTION_RE.findall(text):
        if current_owner:
            facts["members"][function].add(current_owner)
        facts["return_types"].update(_type_names(returns))
        facts["expected_types"].update(_type_names(returns))
        for name, type_name in re.findall(
            r"([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)", params
        ):
            facts["variable_types"][name].add(type_name)
            facts["parameter_types"].add(type_name)
    for name, type_name in PYTHON_ANNOTATED_BINDING_RE.findall(text):
        facts["variable_types"][name].add(type_name)
    for name, type_name in PYTHON_CONSTRUCTOR_BINDING_RE.findall(text):
        facts["variable_types"][name].add(type_name)
    facts["receiver_calls"].update(QUALIFIED_CALL_RE.findall(text))
    for module, names, direct in PYTHON_IMPORT_RE.findall(text):
        facts["imports"].update(_qualified_tokens(module or direct))
        facts["imports"].update(
            token for token in IDENTIFIER_RE.findall(names) if _valid_symbol(token)
        )
    return facts


def _extract_java_structure(text):
    facts = _empty_structure()
    type_matches = list(JAVA_TYPE_RE.finditer(text))
    for match in type_matches:
        owner = match.group(1)
        facts["owners"].add(owner)
        facts["bases"][owner].update(_type_names(match.group(2) or ""))
        facts["bases"][owner].update(_type_names(match.group(3) or ""))
    default_owner = type_matches[-1].group(1) if type_matches else ""

    for match in JAVA_METHOD_RE.finditer(text):
        return_type, method, params = match.groups()
        owner = default_owner
        preceding = [item for item in type_matches if item.start() < match.start()]
        if preceding:
            owner = preceding[-1].group(1)
        if owner:
            facts["members"][method].add(owner)
        return_names = _type_names(return_type)
        facts["return_types"].update(return_names)
        facts["expected_types"].update(return_names)
        for type_name, variable in JAVA_BINDING_RE.findall(params):
            names = _type_names(type_name)
            facts["parameter_types"].update(names)
            facts["variable_types"][variable].update(names)

    for type_name, variable in JAVA_BINDING_RE.findall(text):
        facts["variable_types"][variable].update(_type_names(type_name))
    facts["receiver_calls"].update(QUALIFIED_CALL_RE.findall(text))
    for qualified in JAVA_IMPORT_RE.findall(text):
        facts["imports"].update(_qualified_tokens(qualified))
    for package in JAVA_PACKAGE_RE.findall(text):
        facts["imports"].update(_qualified_tokens(package))
    return facts


def _extract_visible_controls(text, language):
    symbols = set()
    kinds = set()
    lines = (text or "").splitlines()[-100:]
    if (language or "").lower() == "python":
        nonempty = [line for line in lines if line.strip()]
        current_indent = _indent_width(nonempty[-1]) if nonempty else 0
        controls = []
        for line in lines:
            match = PYTHON_CONTROL_RE.match(line)
            if not match:
                continue
            indent = _indent_width(line)
            if indent <= current_indent:
                controls.append((indent, match.group(1), match.group(2)))
        for _, kind, expression in controls[-8:]:
            kinds.add(kind)
            symbols.update(_control_symbols(expression))
    else:
        controls = list(JAVA_CONTROL_RE.finditer("\n".join(lines)))[-8:]
        for match in controls:
            kinds.add(match.group(1).replace(" ", "_"))
            symbols.update(_control_symbols(match.group(2) or ""))
    return {"symbols": symbols, "kinds": kinds}


def _is_related_owner(index, left, right):
    if not left or not right:
        return False
    if left == right:
        return True
    frontier = [left]
    visited = set()
    for _ in range(3):
        next_frontier = []
        for owner in frontier:
            if owner in visited:
                continue
            visited.add(owner)
            related = set(index["bases_by_owner"].get(owner, ()))
            related.update(index["derived_by_base"].get(owner, ()))
            if right in related:
                return True
            next_frontier.extend(related - visited)
        frontier = next_frontier
    return False


def _annotation_names(annotation):
    if annotation is None:
        return set()
    return {
        node.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and _valid_symbol(node.id)
    }


def _annotation_names_many(nodes):
    names = set()
    for node in nodes:
        names.update(_annotation_names(node))
    return names


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _expression_tail(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _type_names(text):
    return {
        token
        for token in IDENTIFIER_RE.findall(text or "")
        if token[:1].isupper() and token not in BUILTIN_TYPES
    }


def _qualified_tokens(value):
    return {
        token
        for token in re.split(r"[^A-Za-z0-9_]+", value or "")
        if _valid_symbol(token) and token != "static"
    }


def _path_tokens(path):
    return {
        token
        for token in re.split(r"[^A-Za-z0-9_]+", (path or "").replace("\\", "/"))
        if _valid_symbol(token)
    }


def _control_symbols(expression):
    calls = {member for _, member in QUALIFIED_CALL_RE.findall(expression or "")}
    return calls | {
        token for token in IDENTIFIER_RE.findall(expression or "") if _valid_symbol(token)
    }


def _valid_symbol(symbol):
    return bool(symbol and len(symbol) > 1 and IDENTIFIER_RE.fullmatch(symbol))


def _indent_width(line):
    prefix = line[: len(line) - len(line.lstrip(" \t"))]
    return len(prefix.expandtabs(4))


def _same_path(left, right):
    return (left or "").replace("\\", "/").lstrip("./") == (
        right or ""
    ).replace("\\", "/").lstrip("./")
