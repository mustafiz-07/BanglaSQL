"""
BanglaSQL — schema grammar for constrained decoding.

Three pieces, in dependency order:

    SchemaIndex   what tables and columns exist, and which table an alias binds to
    parse_prefix  reads partially generated SQL and says what may legally come next
    TokenFilter   turns "one of these identifiers must come next" into a token id list

The point of all three is to let generation be restricted to schema-valid continuations.
Run 7 produced 52 malformed queries out of 351 test examples, and 40 of them named a real
column or table in a position where it was not in scope — not invented identifiers, but
real ones used where nothing bound them.

Why the parser works on decoded text rather than on token ids: BanglaT5 splits English
identifiers into subwords (`course_name` -> ['co','ur','se','_','name']), so token
boundaries carry no syntactic meaning. Decoding a truncated sequence, however, yields
exactly the text prefix ('SELECT c.cour'), which is parseable.
"""

import re

from common import open_readonly_db

# States parse_prefix can report.
TABLE  = "table"    # a table name must come next
COLUMN = "column"   # a column of one specific table must come next
FREE   = "free"     # no identifier constraint applies

IDENT_CHARS = re.compile(r"[A-Za-z0-9_]+")

#: Cache miss marker. `None` is a legitimate cached value here — it means "no token can
#: satisfy this constraint, fail open" — so `dict.get(key)` alone cannot tell a miss from
#: a stored None, and every fail-open lookup would be recomputed, vocabulary scan included.
_MISSING = object()

# Keyword matching is case-sensitive on purpose. Every gold query writes keywords in
# upper case and identifiers in lower case, and `joining_year` is the only identifier that
# begins with a keyword — a case-insensitive rule read "WHERE join|" as the JOIN keyword
# and demanded a table name where `joining_year` was being spelled.
#
# A table name is expected after FROM or JOIN. Group 2 is the separating whitespace and
# group 3 is however much of the name exists. The whitespace may still be absent: the
# space between "FROM" and "students" arrives as part of the '▁students' token, so at the
# moment of the decision the decoded text is "... FROM" with nothing after it.
_AT_TABLE  = re.compile(r"\b(FROM|JOIN)(\s*)([A-Za-z_]*)$")
# A qualified column: single-letter alias, a dot, then however much of the column exists.
# No whitespace is ever pending here, since the dot already closed the boundary.
_AT_COLUMN = re.compile(r"\b([A-Za-z])\.([A-Za-z_]*)$")
# Every alias actually bound by a FROM/JOIN clause, e.g. "FROM students s".
_BOUND     = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]+)\s+([A-Za-z])\b")
# Same clause, alias optional — used to tell which tables are in scope.
_JOINED    = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]+)")
# Every alias referenced by a qualified column anywhere in the text.
_REFERENCED = re.compile(r"\b([A-Za-z])\.[A-Za-z_]+")


class SchemaIndex:
    """The database's tables and columns, plus the corpus's alias convention.

    Every table in the gold corpus aliases to its own first letter and all six first
    letters are distinct (a, c, d, e, i, s), so `c.` resolves to `courses` the moment it
    is emitted — before FROM has been generated. That is what makes it possible to
    constrain a qualified column in the SELECT list at all.
    """

    def __init__(self, tables: dict[str, list[str]]):
        self.tables = tables
        self.table_names = sorted(tables)
        self.all_columns = sorted({c for cols in tables.values() for c in cols})

        self.alias_to_table = {}
        for name in self.table_names:
            letter = name[0].lower()
            # Only trust the convention when it is unambiguous.
            if sum(1 for t in self.table_names if t[0].lower() == letter) == 1:
                self.alias_to_table[letter] = name

    @classmethod
    def from_db(cls, path: str | None = None) -> "SchemaIndex":
        con = open_readonly_db(path) if path else open_readonly_db()
        try:
            tables = {}
            for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                if name.startswith("sqlite_"):
                    continue
                tables[name] = [row[1] for row in con.execute(f"PRAGMA table_info({name})")]
            return cls(tables)
        finally:
            con.close()

    def columns_for_alias(self, alias: str) -> list[str] | None:
        table = self.alias_to_table.get(alias.lower())
        return sorted(self.tables[table]) if table else None


class PrefixState:
    """What the grammar knows after reading the SQL generated so far."""

    __slots__ = ("kind", "names", "partial", "pending", "needs_space")

    def __init__(self, kind, names=(), partial="", pending=frozenset(), needs_space=False):
        self.kind = kind
        self.names = tuple(names)   # identifiers allowed next, when constrained
        self.partial = partial      # the part of the current identifier already emitted
        self.pending = pending      # aliases referenced but not yet bound by FROM/JOIN
        self.needs_space = needs_space  # the word boundary has still to be emitted

    def __repr__(self):
        return (f"PrefixState({self.kind}, partial={self.partial!r}, "
                f"names={len(self.names)}, pending={sorted(self.pending)}, "
                f"needs_space={self.needs_space})")


def pending_aliases(text: str, schema: SchemaIndex) -> frozenset:
    """Aliases used as `x.column` that no FROM/JOIN clause has bound yet.

    This is the obligation that EOS gating enforces. `SELECT c.course_name FROM students s`
    is not wrong yet — `JOIN courses c` may still be coming — so the constraint cannot
    reject the token. It can refuse to let the query end while `c` is still unbound.
    """
    # Only a declared alias counts. `FROM attendance` does not make `a.status` legal —
    # SQLite wants `attendance.status` — and run 7 produced exactly that query six times.
    # No gold query qualifies with an alias its FROM/JOIN clause never declares, so
    # requiring the declaration cannot reject valid SQL.
    bound = {alias.lower() for _, alias in _BOUND.findall(text)}
    used = {a.lower() for a in _REFERENCED.findall(text)}
    return frozenset(a for a in used - bound if a in schema.alias_to_table)


def bound_aliases(text: str, schema: SchemaIndex) -> dict[str, str]:
    """Aliases a FROM/JOIN clause has actually declared, mapped to their table."""
    return {alias.lower(): table.lower()
            for table, alias in _BOUND.findall(text)
            if table.lower() in schema.tables}


def parse_prefix(text: str, schema: SchemaIndex) -> PrefixState:
    """Read partially generated SQL and report what may legally come next."""
    pending = pending_aliases(text, schema)

    match = _AT_TABLE.search(text)
    if match:
        gap, partial = match.group(2), match.group(3)
        return PrefixState(TABLE, schema.table_names, partial, pending,
                           needs_space=(gap == "" and partial == ""))

    match = _AT_COLUMN.search(text)
    if match:
        alias = match.group(1).lower()
        # A declaration already seen wins over the naming convention: if the query said
        # `JOIN students a`, then `a.` means students even though `a` conventionally means
        # attendance. The convention is only a fallback for forward references, where the
        # SELECT list mentions an alias the FROM clause has not reached yet.
        table = bound_aliases(text, schema).get(alias)
        columns = sorted(schema.tables[table]) if table else schema.columns_for_alias(alias)
        if columns:
            return PrefixState(COLUMN, columns, match.group(2), pending)

    return PrefixState(FREE, (), "", pending)


class TokenFilter:
    """Turns "one of these identifiers must come next" into a list of token ids.

    The surface form of a token is read from the raw vocabulary string, not from
    `convert_tokens_to_string`, because that helper strips SentencePiece's '▁' word
    boundary and so reports '▁W' and 'W' identically. They are not interchangeable: the
    first emits ' W' and starts a new word, the second continues the current one. Treating
    them alike blocked `FROM students WHERE` at the W, because 'studentsW' names no table.

    Carrying the space inside the surface makes the test uniform — word-initial and
    continuation tokens are both just string prefixes of what remains to be spelled.

    Answers are cached on (identifiers, partial, needs_space). The hook fires ~112k times
    per test run, so without the cache this would dominate evaluation time.
    """

    #: A token can only take part in spelling an identifier if it looks like one,
    #: optionally preceded by the single space that opens a new word.
    _USABLE = re.compile(r" ?[A-Za-z0-9_]+")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        special = set(tokenizer.all_special_ids) - {self.eos_id}

        # The two sets overlap on purpose. '▁W' can spell the start of a new identifier
        # (when a word boundary is still pending) and can equally close the identifier just
        # finished, because the space it carries ends that word. Filing it only as a
        # spelling token is what blocked `FROM students WHERE` at the W.
        self.surfaces = []     # (token_id, surface) for tokens that can spell an identifier
        self.terminators = []  # tokens that may follow a completed identifier
        for token, token_id in tokenizer.get_vocab().items():
            if token_id in special:
                continue
            surface = token.replace("▁", " ")
            if not surface:
                continue
            if self._USABLE.fullmatch(surface):
                self.surfaces.append((token_id, surface))
            if not (surface[0].isalnum() or surface[0] == "_"):
                self.terminators.append(token_id)
        self.terminators.append(self.eos_id)

        self.all_ids = sorted(tokenizer.get_vocab().values())
        self.all_but_eos = [i for i in self.all_ids if i != self.eos_id]
        self._cache = {}

    def allowed(self, names: tuple, partial: str, needs_space: bool = False,
                allow_eos: bool = True) -> list[int] | None:
        """Token ids that carry `partial` further toward one of `names`.

        `allow_eos=False` withholds end-of-sequence even where the identifier is complete.
        The caller needs that because a finished identifier admits terminators, and
        end-of-sequence is one of them — so `SELECT c.course_name FROM courses` could stop
        there with `c` still unbound, which is the very thing EOS gating exists to prevent.

        Returns None when nothing can satisfy the constraint, which the caller treats as
        "fall back to unconstrained" — a grammar bug must degrade to current behaviour,
        never stall generation.
        """
        key = (names, partial, needs_space, allow_eos)
        cached = self._cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached

        # What still has to be spelled, including the pending word boundary.
        lead = " " if needs_space else ""
        remainders = [lead + n[len(partial):] for n in names if n.startswith(partial)]
        if not remainders:
            self._cache[key] = None
            return None

        allowed = [tid for tid, surface in self.surfaces
                   if any(r.startswith(surface) for r in remainders)]
        # A finished identifier may be followed by punctuation or a space, so generation is
        # not trapped inside a name it has already spelled out.
        if partial in names and not needs_space:
            allowed.extend(self.terminators if allow_eos
                           else (t for t in self.terminators if t != self.eos_id))

        result = sorted(set(allowed)) or None
        self._cache[key] = result
        return result
