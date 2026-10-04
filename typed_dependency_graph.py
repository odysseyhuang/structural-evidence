import ast
import re
from collections import defaultdict


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
QUALIFIED_CALL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\("
)
PYTHON_DEF_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
PYTHON_CLASS_RE = re.compile(r"(?m)^\s*class\s+([A-Za-z_]\w*)\b")
PYTHON_ASSIGN_RE = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*(?::[^=\n]+)?=")
JAVA_TYPE_DEF_RE = re.compile(
    r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
JAVA_METHOD_DEF_RE = re.compile(
    r"(?m)^\s*(?:(?:public|protected|private|static|final|abstract|"
    r"synchronized|native|default|strictfp)\s+)*"
    r"(?:<[^>{};]+>\s+)?(?:[A-Za-z_$][\w$<>,.?\[\]]*\s+)+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^;{}]*\)\s*"
    r"(?:throws\s+[^{]+)?(?:\{|$)"
)
JAVA_VARIABLE_DEF_RE = re.compile(
    r"\b(?:final\s+)?([A-Za-z_$][\w$<>,.?\[\]]*|byte|short|int|long|"
    r"float|double|boolean|char)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?==|;|,)"
)
JAVA_NEW_TYPE_RE = re.compile(r"\bnew\s+([A-Za-z_$][A-Za-z0-9_$.]*)")
JAVA_EXTENDS_TYPE_RE = re.compile(
    r"\b(?:extends|implements|instanceof|throws|catch)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$.]*)"
)

KEYWORDS = {
    "False", "None", "True", "abstract", "and", "as", "assert", "async",
    "await", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "def", "default", "del", "do", "double", "elif",
    "else", "enum", "except", "extends", "final", "finally", "float", "for",
    "from", "global", "goto", "if", "implements", "import", "in", "instanceof",
    "int", "interface", "is", "lambda", "long", "native", "new", "nonlocal",
    "not", "null", "or", "package", "pass", "private", "protected", "public",
    "raise", "record", "return", "self", "short", "static", "strictfp", "super",
    "switch", "synchronized", "this", "throw", "throws", "transient", "try",
    "true", "void", "volatile", "while", "with", "yield",
}

WEAK_SYMBOLS = {
    "arg", "args", "cfg", "config", "data", "item", "items", "key", "kwargs",
    "obj", "option", "options", "param", "params", "result", "results", "self",
    "test", "tests", "tmp", "value", "values",
}

RELATION_PRIORITY = {
    "graph_typed_type": 0,
    "graph_typed_call": 1,
    "graph_typed_def_use": 2,
}


class TypedDependencyIndex:
    """A lightweight, definition-backed dependency index over benchmark code blocks."""

    def __init__(self, code_blocks_by_task):
        self.code_blocks_by_task = code_blocks_by_task
        self.facts_by_task = {}
        self.definition_index_by_task = {}

    def neighbors(
        self,
        task_id,
        source_text,
        language,
        max_results,
        max_df,
        weights,
        query_bonus=0.0,
        origin="seed",
        exclude_block_key=None,
        exclude_file_path="",
        allowed_relations=None,
    ):
        if max_results <= 0:
            return []

        self._ensure_task(task_id)
        facts = extract_dependency_facts(source_text, language)
        definition_index = self.definition_index_by_task.get(task_id, {})
        code_blocks = self.code_blocks_by_task.get(task_id, [])
        proposals = {}

        requests = (
            ("graph_typed_type", facts["types"], {"type"}),
            ("graph_typed_call", facts["calls"], {"callable"}),
            (
                "graph_typed_def_use",
                facts["uses"],
                {"type", "callable"},
            ),
        )
        for relation, symbols, required_kinds in requests:
            if allowed_relations is not None and relation not in allowed_relations:
                continue
            for symbol in sorted(symbols):
                entries = definition_index.get(symbol, [])
                if not entries or len(entries) > max_df:
                    continue
                for idx, kinds in entries:
                    if required_kinds and not required_kinds.intersection(kinds):
                        continue
                    block = code_blocks[idx]
                    if exclude_block_key is not None and block_key(block) == exclude_block_key:
                        continue
                    if exclude_file_path and _same_path(block.file_path, exclude_file_path):
                        continue

                    score = float(weights.get(relation, 1.0))
                    score += 1.0 / max(1, len(entries))
                    if origin == "query":
                        score += max(0.0, float(query_bonus))

                    previous = proposals.get(idx)
                    if previous is None or _proposal_key(score, relation) < _proposal_key(
                        previous["score"], previous["relation"]
                    ):
                        proposals[idx] = {
                            "block": block,
                            "relation": relation,
                            "score": score,
                            "symbols": {symbol},
                            "origin": origin,
                        }
                    elif previous["relation"] == relation:
                        previous["symbols"].add(symbol)

        ranked = sorted(
            proposals.values(),
            key=lambda item: (
                -item["score"],
                RELATION_PRIORITY.get(item["relation"], 99),
                block_key(item["block"]),
            ),
        )
        for item in ranked:
            item["symbols"] = tuple(sorted(item["symbols"]))
        return ranked[:max_results]

    def _ensure_task(self, task_id):
        if task_id in self.definition_index_by_task:
            return

        facts_by_block = []
        definition_index = defaultdict(list)
        for idx, block in enumerate(self.code_blocks_by_task.get(task_id, [])):
            facts = extract_dependency_facts(
                getattr(block, "code_content", ""),
                getattr(block, "language", ""),
            )
            facts_by_block.append(facts)
            for symbol, kinds in facts["definitions"].items():
                definition_index[symbol].append((idx, frozenset(kinds)))

        self.facts_by_task[task_id] = facts_by_block
        self.definition_index_by_task[task_id] = dict(definition_index)


def extract_dependency_facts(text, language):
    if (language or "").lower() == "python":
        return _extract_python_facts(text or "")
    return _extract_java_facts(text or "")


def block_key(block):
    return (
        getattr(block, "file_path", ""),
        getattr(block, "description", ""),
        getattr(block, "code_content", ""),
    )


class _PythonDependencyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.definitions = defaultdict(set)
        self.calls = set()
        self.types = set()
        self.uses = set()

    def visit_FunctionDef(self, node):
        self.definitions[node.name].add("callable")
        for argument in list(node.args.args) + list(node.args.kwonlyargs):
            self.definitions[argument.arg].add("value")
            self.types.update(_annotation_names(argument.annotation))
        if node.args.vararg:
            self.definitions[node.args.vararg.arg].add("value")
        if node.args.kwarg:
            self.definitions[node.args.kwarg.arg].add("value")
        self.types.update(_annotation_names(node.returns))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.definitions[node.name].add("type")
        for base in node.bases:
            self.types.update(_annotation_names(base))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.definitions[node.id].add("value")
        elif isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)

    def visit_Call(self, node):
        symbol = _python_callable_name(node.func)
        if symbol:
            self.calls.add(symbol)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.types.update(_annotation_names(node.annotation))
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.definitions[alias.asname or alias.name.split(".")[0]].add("value")

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.definitions[alias.asname or alias.name].add("value")


def _extract_python_facts(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _extract_python_facts_regex(text)

    visitor = _PythonDependencyVisitor()
    visitor.visit(tree)
    return _clean_facts(visitor.definitions, visitor.calls, visitor.types, visitor.uses)


def _extract_python_facts_regex(text):
    definitions = defaultdict(set)
    for symbol in PYTHON_DEF_RE.findall(text):
        definitions[symbol].add("callable")
    for symbol in PYTHON_CLASS_RE.findall(text):
        definitions[symbol].add("type")
    for symbol in PYTHON_ASSIGN_RE.findall(text):
        definitions[symbol].add("value")

    calls = {_tail_symbol(match) for match in QUALIFIED_CALL_RE.findall(text)}
    types = {token for token in IDENTIFIER_RE.findall(text) if token[:1].isupper()}
    uses = set(IDENTIFIER_RE.findall(text))
    return _clean_facts(definitions, calls, types, uses)


def _extract_java_facts(text):
    definitions = defaultdict(set)
    for symbol in JAVA_TYPE_DEF_RE.findall(text):
        definitions[symbol].add("type")
    for symbol in JAVA_METHOD_DEF_RE.findall(text):
        definitions[symbol].add("callable")
    for type_name, symbol in JAVA_VARIABLE_DEF_RE.findall(text):
        definitions[symbol].add("value")

    calls = {_tail_symbol(match) for match in QUALIFIED_CALL_RE.findall(text)}
    calls.difference_update(
        symbol for symbol, kinds in definitions.items() if "callable" in kinds
    )

    types = set()
    for type_name, _ in JAVA_VARIABLE_DEF_RE.findall(text):
        types.update(_java_type_symbols(type_name))
    for type_name in JAVA_NEW_TYPE_RE.findall(text):
        types.add(_tail_symbol(type_name))
    for type_name in JAVA_EXTENDS_TYPE_RE.findall(text):
        types.add(_tail_symbol(type_name))
    types.update(
        token for token in IDENTIFIER_RE.findall(text) if token[:1].isupper()
    )

    uses = set(IDENTIFIER_RE.findall(text))
    return _clean_facts(definitions, calls, types, uses)


def _clean_facts(definitions, calls, types, uses):
    clean_definitions = {}
    for symbol, kinds in definitions.items():
        if _valid_symbol(symbol, allow_weak=True):
            clean_definitions[symbol] = set(kinds)

    clean_calls = {
        symbol for symbol in calls
        if _valid_symbol(symbol) and symbol not in clean_definitions
    }
    clean_types = {
        symbol for symbol in types
        if _valid_symbol(symbol, allow_weak=True)
    }
    clean_uses = {
        symbol for symbol in uses
        if _valid_symbol(symbol) and symbol not in clean_definitions
    }
    clean_uses.difference_update(clean_calls)
    clean_uses.difference_update(clean_types)
    return {
        "definitions": clean_definitions,
        "calls": clean_calls,
        "types": clean_types,
        "uses": clean_uses,
    }


def _annotation_names(annotation):
    if annotation is None:
        return set()
    return {
        node.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and _valid_symbol(node.id, allow_weak=True)
    }


def _python_callable_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _java_type_symbols(type_name):
    return {
        token for token in IDENTIFIER_RE.findall(type_name)
        if token[:1].isupper() and _valid_symbol(token, allow_weak=True)
    }


def _valid_symbol(symbol, allow_weak=False):
    if not symbol or len(symbol) <= 1 or symbol in KEYWORDS:
        return False
    if not allow_weak and symbol in WEAK_SYMBOLS:
        return False
    return bool(IDENTIFIER_RE.fullmatch(symbol))


def _tail_symbol(symbol):
    return (symbol or "").split(".")[-1]


def _same_path(left, right):
    return (left or "").replace("\\", "/").lstrip("./") == (
        right or ""
    ).replace("\\", "/").lstrip("./")


def _proposal_key(score, relation):
    return (-score, RELATION_PRIORITY.get(relation, 99))
