# Schema-Constrained Decoding — implementation plan

Branch: `model_contribution`. Inference-only: no retraining, no change to the training
data or the model weights, so the ablation is the same checkpoint decoded two ways.

## Why

Classification of all 52 malformed queries in run 7's test predictions
(`logs_archive (5)/test_predictions.json`):

| failure | n | mechanism that fixes it |
|---|---|---|
| real column, owning table **never joined** | 22 | EOS gating on pending aliases |
| real column emitted in **table position** (`JOIN joining_year`) | 12 | table-position constraint |
| invented / alias-like name | 8 | identifier constraint (largely gone in run 8) |
| real column, table joined, **wrong alias** | 6 | qualified-column constraint |
| ambiguous unqualified column | 3 | out of scope for this pass |
| syntax error | 1 | — |

The addressable set is **52 of 351 test examples (14.8%)**, and that figure is exact
rather than estimated: `pick_executable` already reranks to any executable beam, so a
*chosen* prediction being malformed means every one of the four beams failed. These are
precisely the cases reranking cannot reach.

Expected effect, stated honestly:

- **validity rate** 85.2% → 93–97%. Near-certain; constrained decoding makes
  schema-invalid output unreachable.
- **execution accuracy** +4 to +10 points. Blocking a malformed query only forces the
  model to emit something else, which may still be semantically wrong. The gain comes
  only where the next-best schema-valid continuation happens to be correct.

## Two facts the design rests on, both verified

**1. Aliases are deterministic.** Every table in the gold corpus aliases to its own first
letter, and all six first letters are distinct (a, c, d, e, i, s). So `c.` means `courses`
the moment it is emitted, before `FROM` has been generated.

**2. Partial decoding is clean.** BanglaT5 shreds identifiers into subwords
(`course_name` → `['co','ur','se','_','name']`), but decoding a truncated token sequence
yields exactly the text prefix — `SELECT c.cour` — so the parser can work on decoded text
rather than on token boundaries.

## The ordering problem, and how it is solved

SQL writes `SELECT c.course_name` before `FROM courses c`. When the decoder emits `c.` it
has not yet said what `c` binds to, so scope cannot be validated left to right. This is
exactly the 22-error class.

Rather than blocking the token, the constraint records `c` as a **pending obligation** and
refuses the end-of-sequence token while any obligation is unmet. The model may write
`c.course_name` — it simply may not *finish* until `courses` has been joined.

## Design

```
per beam, per decoding step:
  1. decode tokens-so-far to text
  2. parse the tail of that text into a state:
       ... FROM|JOIN <partial>        -> TABLE   (allowed = the 6 table names)
       ... <alias>.<partial>          -> COLUMN  (allowed = columns of alias's table)
       anything else                  -> FREE
     and collect pending aliases = referenced in `x.` minus bound in FROM/JOIN
  3. TABLE/COLUMN -> allow tokens t where partial + text(t) prefixes an allowed name;
                     if partial is already a complete name, also allow terminators
     FREE          -> allow everything, minus EOS while obligations are pending
```

**Modules**

| file | role |
|---|---|
| `schema_grammar.py` | `SchemaIndex` read from the live DB, prefix parser, token filtering |
| `constrained_decode.py` | `build_prefix_fn(tokenizer, schema)` — the closure `generate()` calls |
| `evaluate.py` | `--constrained` flag; reports both decoders side by side |

**Performance.** The hook fires roughly 112,000 times per test run (351 examples × 4 beams
× ~80 steps). Only 2,659 of the 32,100 vocabulary tokens are pure identifier characters,
so the candidate scan is small, and results are cached on `(allowed set, partial)` — after
warm-up almost every call is a dict lookup.

**Safety valves.** An empty allowed set falls back to unconstrained. EOS is gated only
while there is room left before `max_length`, so a stuck beam can always terminate, and
`pick_executable` remains the final net.

## Phases

1. `SchemaIndex` + token filtering — *verify:* every table and column reachable.
2. Prefix parser — *verify:* replayed over every gold query, prefix by prefix.
3. Table-position constraint — *verify (critical gate):* force-decode all 228 gold queries
   under the constraint; none may be blocked. A constraint that rejects valid SQL is worse
   than no constraint.
4. Qualified-column constraint + EOS gating — *verify:* same gate, plus replay of run 7's
   52 malformed predictions to confirm each would have been blocked.
5. `evaluate.py --constrained`, benchmark, ablation table.

Phase 6 (ambiguous-column qualification, 3 cases) is deliberately excluded: lowest gain,
highest risk of blocking valid gold.

---

## Verification (all phases complete, ablation pending a checkpoint)

| gate | result |
|---|---|
| gold queries blocked, character level | 0 / 228 |
| gold queries blocked, token level | **0 / 228** |
| gold queries ending with an unbound alias | 0 / 228 |
| run 7 malformed predictions caught | **41 / 52 (79%)** |
| — by identifier constraint | 18 |
| — by EOS gating on an unbound alias | 23 |
| constraint failed open (grammar gap) | 0 |
| overhead | 0.09 ms/step, ~0.2 min per test run |

The 11 uncaught are 8 bare aggregate aliases (`avg_grade_point`), which run 8 removed
from the gold so the model has no reason to emit them, and the 3 ambiguous-column cases
deliberately left out of scope.

Selectivity spot check — after `SELECT * FROM`, 14 of 32,100 tokens remain, exactly those
that can begin one of the six table names; after an unbound `c.`, end-of-sequence is
refused.

## Two bugs the gates caught before any GPU time

**1. SentencePiece boundary stripping.** `convert_tokens_to_string` discards the '▁' word
marker, so '▁W' and 'W' both report as "W" although the first emits " W" and starts a new
word. Treating them alike blocked 186 of 228 gold queries — `FROM students WHERE` died at
the W, because "studentsW" names no table. Fixed by reading the raw vocabulary token and
translating '▁' to a space, so the surface carries its own boundary.

**2. A keyword that is also an identifier prefix.** Matching FROM/JOIN case-insensitively
read `WHERE join|` as the JOIN keyword and demanded a table name where `joining_year` was
being spelled — 9 gold queries blocked. Gold writes keywords in upper case and identifiers
in lower case, and `joining_year` is the only identifier beginning with a keyword, so
case-sensitive matching resolves it.

Both were invisible in the error analysis and would have shown up only as a mysterious
accuracy drop. They are the reason the force-decode gate is phase 3's exit criterion.

## Running the ablation

Same checkpoint, two decoders — no retraining:

```bash
python evaluate.py                  # baseline: execution-guided reranking only
python evaluate.py --constrained    # + schema-constrained generation
```

Compare `validity_rate` first (the direct claim), then `execution_accuracy` and the
`malformed_sql` row of the failure table.
