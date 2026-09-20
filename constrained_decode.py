"""
BanglaSQL — schema-constrained decoding.

Execution-guided decoding (Wang et al., 2018, implemented in common.pick_executable)
generates a full beam and then discards the ones that do not run. That cannot help when
*every* beam is invalid, which is exactly what run 7's 52 malformed test predictions were:
`pick_executable` had already reranked and still had nothing valid to return.

This module constrains generation instead of filtering it, by masking the token
distribution at each step so that only schema-valid continuations remain reachable.

Three constraints, matched to the three failure classes measured in run 7:

    table position    after FROM/JOIN only a real table name may be spelled     12 cases
    qualified column  after `c.` only a column of `courses` may be spelled      10 cases
    EOS gating        the query may not end while an alias is still unbound     22 cases

The third is the one that needs explaining. SQL writes `SELECT c.course_name` before
`FROM courses c`, so when the decoder emits `c.` it has not yet said what `c` binds to and
the reference cannot be judged wrong yet. So the constraint does not reject the token; it
records an obligation and refuses end-of-sequence until `courses` has been joined.

Every constraint fails open. An empty allowed set, or a beam close to max_length, returns
the unconstrained vocabulary — a grammar bug degrades to current behaviour rather than
stalling generation.
"""

from schema_grammar import FREE, SchemaIndex, TokenFilter, parse_prefix

# EOS stays available near the length limit, so a beam that cannot satisfy its obligations
# can still terminate instead of running to max_length and emitting noise.
EOS_GATE_MARGIN = 16


class ConstrainedDecoder:
    """Builds the `prefix_allowed_tokens_fn` that `model.generate` calls per beam, per step."""

    def __init__(self, tokenizer, schema: SchemaIndex | None = None,
                 max_length: int = 160, eos_gating: bool = True):
        self.tokenizer = tokenizer
        self.schema = schema or SchemaIndex.from_db()
        self.filter = TokenFilter(tokenizer)
        self.max_length = max_length
        self.eos_gating = eos_gating
        self.stats = {"calls": 0, "identifier_constrained": 0, "eos_blocked": 0,
                      "fallback_empty": 0}

    def allowed_tokens(self, token_ids) -> list[int]:
        """The token ids that may legally follow the sequence generated so far.

        The two constraints are combined rather than branched between. Returning early
        from the identifier branch silently skipped the EOS gate, and since a completed
        identifier admits terminators — end-of-sequence among them — a beam sitting at
        `SELECT c.course_name FROM courses` could stop with `c` still unbound. That is the
        exact failure the gate was written to prevent, so the gate has to be decided first
        and then applied to whichever candidate set the state produced.
        """
        self.stats["calls"] += 1
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        state = parse_prefix(text, self.schema)

        # EOS is withheld while an alias is still unbound, unless the beam is close enough
        # to max_length that it needs a way out.
        block_eos = bool(self.eos_gating and state.pending
                         and len(token_ids) < self.max_length - EOS_GATE_MARGIN)
        if block_eos:
            self.stats["eos_blocked"] += 1

        if state.kind != FREE:
            allowed = self.filter.allowed(state.names, state.partial, state.needs_space,
                                          allow_eos=not block_eos)
            if allowed:
                self.stats["identifier_constrained"] += 1
                return allowed
            # Nothing can satisfy the constraint — the model is somewhere the grammar does
            # not model. Fail open rather than force a wrong identifier.
            self.stats["fallback_empty"] += 1

        return self.filter.all_but_eos if block_eos else self.filter.all_ids

    def prefix_fn(self):
        """The callable `generate(prefix_allowed_tokens_fn=...)` expects."""
        def fn(batch_id, input_ids):
            return self.allowed_tokens(input_ids.tolist())
        return fn

    def report(self) -> str:
        s = self.stats
        if not s["calls"]:
            return "constrained decoding: not invoked"
        return (f"constrained decoding: {s['calls']} steps, "
                f"{s['identifier_constrained']} identifier-constrained "
                f"({s['identifier_constrained'] / s['calls']:.1%}), "
                f"{s['eos_blocked']} EOS-blocked, "
                f"{s['fallback_empty']} failed open")
