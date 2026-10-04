import os
import re
from collections import OrderedDict
from dataclasses import dataclass


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
API_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+")

PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false", "null",
}

COMMON_STOPWORDS = {
    "self", "cls", "var", "val", "args", "kwargs", "str", "int", "float",
    "list", "dict", "set", "tuple", "len", "range", "print", "return",
}


@dataclass(frozen=True)
class QueryView:
    source: str
    query: str


def build_base_query(example, context_len=20):
    return "\n".join([x for x in example.left_context.split("\n") if x.strip() != ""][-context_len:])


def build_query_bundle(args, example, base_query=None, draft_prediction=None, context_len=20):
    """
    Builds leakage-safe query views from left context and file path only.
    draft_prediction may be supplied by the existing RepoCoder first-pass generation.
    """
    if base_query is None:
        base_query = build_base_query(example, context_len=context_len)

    views = [QueryView("base", base_query)]

    if not getattr(args, "ucm_disable_identifier_query", False):
        identifier_query = build_identifier_query(
            example.left_context,
            getattr(args, "ucm_query_identifier_limit", 64),
        )
        if identifier_query:
            views.append(QueryView("identifier", identifier_query))

    if not getattr(args, "ucm_disable_import_api_query", False):
        import_api_query = build_import_api_query(
            example.left_context,
            getattr(args, "ucm_query_import_limit", 32),
        )
        if import_api_query:
            views.append(QueryView("import_api", import_api_query))

    path_query = build_path_query(example.file_path)
    if getattr(args, "ucm_enable_path_query", False) and path_query:
        views.append(QueryView("path", path_query))

    if draft_prediction:
        draft_query = "\n".join([base_query, draft_prediction]).strip()
        if draft_query:
            views.append(QueryView("draft", draft_query))

    return views


def build_identifier_query(left_context, limit):
    tokens = []
    seen = OrderedDict()
    keyword_set = PYTHON_KEYWORDS | JAVA_KEYWORDS | COMMON_STOPWORDS

    for token in IDENTIFIER_RE.findall(left_context):
        if token in keyword_set or len(token) <= 1:
            continue
        if token not in seen:
            seen[token] = None
            tokens.append(token)

    return " ".join(tokens[-limit:])


def build_import_api_query(left_context, limit):
    import_lines = []
    api_tokens = []
    seen_api = OrderedDict()

    for line in left_context.splitlines():
        stripped = line.strip()
        if _is_import_line(stripped):
            import_lines.append(stripped)
        for token in API_RE.findall(stripped):
            if token not in seen_api:
                seen_api[token] = None
                api_tokens.append(token)

    parts = import_lines[-limit:] + api_tokens[-limit:]
    return "\n".join(parts[-limit:])


def build_path_query(file_path):
    if not file_path:
        return ""

    normalized = file_path.replace("\\", "/")
    path_parts = [x for x in re.split(r"[/._\-\s]+", normalized) if x]
    filename = os.path.basename(normalized)
    stem, ext = os.path.splitext(filename)

    tokens = []
    for part in path_parts + [stem, ext.lstrip(".")]:
        if not part:
            continue
        tokens.append(part)
        tokens.extend(CAMEL_RE.findall(part))

    deduped = []
    seen = set()
    for token in tokens:
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)

    return " ".join(deduped)


def _is_import_line(stripped_line):
    if not stripped_line:
        return False
    return (
        stripped_line.startswith("import ")
        or stripped_line.startswith("from ")
        or stripped_line.startswith("package ")
    )
