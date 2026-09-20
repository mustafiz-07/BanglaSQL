# BanglaSQL — Run-by-Run Progress Log

Bangla natural-language questions → SQL, over a fixed 6-table University Management
System database (SQLite: `departments`, `instructors`, `students`, `courses`,
`enrollments`, `attendance`; 304 students, 471 enrollments, 500 attendance rows).

Each run below records three things: **what it had** (dataset construction + model
setup), **what it produced** (measured results), and **where it failed and why**.
Every run after the first opens by stating which of the previous run's failures it
addresses.

**Primary metric is execution accuracy** — the generated SQL is run against the real
database and its result set compared to the gold query's. Row order is compared only
when the gold query has an `ORDER BY`. Exact match (string equality after
normalisation) is reported as a secondary metric because it rejects SQL that is
correct but written with a different alias or column order.

---

## Summary

| run | key change | test exec. acc. | exact match | validity |
|---|---|---|---|---|
| 0 | baseline as originally written | *not measurable* | — | — |
| 1 | fixed the train/dev/test split; built the evaluation harness | 27.2% | 21.7% | 63.3% |
| 2 | value-slot augmentation, question-only input, execution-guided decoding | 45.5% | 23.5% | 81.8% |
| 3 | +63 templates for under-covered query types, LR 2e-4 | 41.8% | 34.0% | 80.8% |
| 4 | polarity augmentation, stable split | 38.0% | 21.8% | 83.2% |
| 5 | duplicate-column rejection, gold-query audit (+ trained 10 epochs longer) | **58.7%** | 34.8% | 91.7% |
| 6 | patience 5→10, orphan ASC/DESC rejection, run-on phrasing repair | 54.6% | 34.8% | 90.9% |
| 7 | **dataset rebuilt** — 13 question/SQL errors fixed, database repopulated, every loophole closed | 55.0% | **50.4%** | 85.2% |
| 8 | আগে/পরে polarity rule, aggregate aliases removed, target length 128→160 | *pending* | | |

**Headline numbers are only comparable from run 4 onward.** Runs 0→4 each rebuilt the
test set, so the exec-accuracy column above tracks four different test sets. Runs 3 and
4 share only 4 of 34 test templates. From run 5 on, the more important caveat is
**variance**: the test set is 339 examples but only **34 distinct templates**, so a
cluster bootstrap over templates gives run 5 a 95% CI of 0.40–0.76 and run 6 one of
0.37–0.71. Differences smaller than about 10 points are not measurable here. Component scores, failure categories and
within-run comparisons are the meaningful readings until run 5.

---

## Run 0 — baseline (as originally written)

### What it had

**Dataset.** 165 hand-written Bangla question–SQL templates (65 easy, 100 medium)
across 28 `query_type` categories. Augmentation was paraphrase-only — synonym
substitution plus prefix/suffix "register frames" — inflating the corpus roughly 15×
to 2,475 pairs, split 1381 / 577 / 517.

The split held out **entire query types**: a whole category was assigned wholesale to
dev or test.

**Model.** BanglaT5 (`csebuetnlp/banglat5`, 247,577,856 parameters), a T5-style
encoder-decoder. Input was the question with the linearised schema appended:

```
translate Bangla to SQL: <বাংলা প্রশ্ন> </s> table: departments(dept_id, ...) | table: students(...)
```

Batch 8 × 2 gradient accumulation, LR 2e-4, up to 20 epochs, early stopping on dev
exact match (patience 5, 6-epoch warm-up), beam search (4 beams) at evaluation.

### What it produced

No test results exist for this run — **there was no evaluation harness**. The project
plan named execution accuracy as the primary metric but nothing implemented it.
Training reported dev exact match rising to 0.2704 by epoch 8.

### Where it failed, and why

**1. The split made the task partly impossible.** Train covered 19 of 28 query types.
Dev contained 6 types, **5 of which never appeared in training**; test contained 5
types, **4 unseen**.

| split | examples | query types | types absent from train |
|---|---|---|---|
| train | 1381 | 19 | — |
| dev | 577 | 6 | `select_all`, `order_by`, `order_by_limit`, `aggregate_group`, `aggregate_group_order` |
| test | 517 | 5 | `join_where`, `limit`, `select_column`, `select_distinct` |

The model was graded on `SELECT * FROM students`, `ORDER BY`, `LIMIT` and `DISTINCT`
— shapes it was never shown. Worse, dev exact match drove **both** early stopping and
best-checkpoint selection, so training was steered by a metric that was near-zero by
construction. Holding out whole query patterns is a legitimate design at Spider scale;
with 165 templates on one fixed schema it just removes the ability to learn.

**2. Augmentation added almost no linguistic diversity.** The 15× inflation came
mostly from tacking prefixes on: 93 training questions began `SQL দিয়ে বের করো:` and
96 ended with an ungrammatical ` কী?` (*"...গ্রেড দেখাও কী?"*). Effective semantic
diversity stayed at 165.

**3. Some synonym substitutions changed meaning or register.** শিক্ষার্থী → শিক্ষানবিস
(*apprentice*, not *student*), গ্রেড → মার্ক/রেজাল্ট, নামতা ক্রমে (*multiplication
table*) for descending order, and Sadhu forms (ভর্তি হইয়াছে) that the project scope
explicitly excludes.

**4. No execution accuracy, no demo.** Phases 4 and 5 of the plan were unimplemented.

What was *not* wrong: the database and gold SQL were sound — all 165 queries execute,
only 4 return zero rows — and BanglaT5 was the right model choice.

---

## Run 1 — fix the split, build the evaluation harness

> **Addresses run 0's failures:** split redesigned so no query type is unlearnable (1);
> augmentation cleaned and cut (2, 3); execution-accuracy harness and Streamlit demo
> written (4).

### What it had

**Dataset.** Same 165 templates. Augmentation reduced from ~15× to 6× (990 pairs) by
removing the meaning-changing synonyms, deleting the ungrammatical ` কী?` frame and the
`SQL দিয়ে বের করো:` prefix, and making register frames sentence-type aware — polite
imperatives (অনুগ্রহ করে) only attach to commands ending in "।", never to a "কত?"
question.

**Split redesigned to template-level holdout, stratified by `query_type`.** Templates
are partitioned *within* each query type, and every augmented variant follows its base
template. This gives two properties the old split lacked:

- train contains at least one template of **every** query type, so no shape is
  unreachable;
- dev/test templates are entirely unseen — new SQL *and* new Bangla — so the score
  measures generalisation, not memorised paraphrases.

| | train | dev | test |
|---|---|---|---|
| examples | 642 | 168 | 180 |
| templates | 107 | 28 | 30 |
| distinct SQL queries | 107 | — | — |

All 28 query types present in train; 0 test SQL queries seen during training.

**Model.** Unchanged architecture. Input still carried the schema, but the stray
literal `</s>` was removed and prompt construction was centralised in `common.py` so
training, evaluation and the demo cannot drift apart. LR 3e-4, batch 8 (no
accumulation), up to 40 epochs, early stopping on dev exact match (patience 6,
8-epoch warm-up).

**New: `evaluate.py`.** Execution accuracy, exact match, validity rate, per-component
accuracy (SELECT / WHERE / GROUP BY / HAVING / ORDER BY / JOIN / tables / aggregates),
breakdowns by difficulty and query type, and a failure-category table. The database is
opened read-only and non-`SELECT` statements are rejected.

### What it produced

Best epoch 7, early-stopped at 13. Training wall time ~41 min on a T4.

| metric | test (180 examples) |
|---|---|
| Execution accuracy | **27.2%** |
| Exact match | 21.7% |
| Validity rate | 63.3% |

By difficulty: easy 33.3%, medium 23.2%. Component accuracy: tables 71.1%, join 57.7%,
group_by 57.6%, aggregate 53.5%, select 43.3%, where 42.9%, order_by 28.6%, having 7.1%.

### Where it failed, and why

**1. The model memorised instead of composing.** Train loss fell to 0.011 while dev
loss bottomed at epoch 3 (0.2823) and rose steadily to 0.4006. Classic overfitting.

**Why:** paraphrasing changes only the Bangla. All 107 training templates mapped to
**107 distinct SQL strings**, each seen ~6 times with near-identical wording. The model
learned to recall those strings; the test set asks for SQL it has never produced.

**2. 36.7% of outputs did not execute** (66 of 180). Of those, **49 were `no such
column`** — schema-grounding errors, not syntax errors:

| gold | predicted |
|---|---|
| `... FROM students s JOIN enrollments e ... WHERE e.grade = 'A+'` | `SELECT first_name, last_name FROM students WHERE grade = 'A+'` (join dropped) |
| `SELECT DISTINCT status FROM attendance` | `SELECT DISTINCT designation FROM attendance` |
| `... WHERE status = 'Absent'` | `... WHERE absent_count = 'Absent'` (invented column) |

The model put columns on the wrong table and invented plausible names — consistent
with recalling memorised fragments rather than reading the schema.

**3. The tokenizer was ruled out as a cause.** All 165 gold queries encode and decode
back unchanged under the BanglaT5 tokenizer, and no SQL character maps to `<unk>`.

---

## Run 2 — make the model compose, and stop wasting input

> **Addresses run 1's failures:** value-slot augmentation creates new SQL targets so
> recall is no longer sufficient (1); the constant schema string is removed and
> execution-guided decoding recovers unexecutable top beams (2).

### What it had

**Dataset — value-slot augmentation added.** A literal is swapped **consistently in the
question and the SQL**: department (গণিত ↔ `'Mathematics'`), CGPA / grade-point / year
thresholds (৩.৫ ↔ `3.5`), grade, semester, course name, `LIMIT`. A variant is kept only
if its SQL executes and returns rows, and only if that SQL does not already exist
anywhere — which preserves the zero-leakage guarantee. Each variant follows its base
template into the same split.

This attacks run 1's root cause directly: it multiplies distinct **targets**, not just
surface forms, and forces the model to copy values out of the question.

| | train | dev | test |
|---|---|---|---|
| examples | 993 | 213 | 264 |
| templates | 107 | 28 | 30 |
| distinct SQL queries | **224** (was 107) | 43 | 58 |

**Model — schema removed from the input.**

```
translate Bangla to SQL: <বাংলা প্রশ্ন>
```

With a single fixed database the linearised schema is identical in every example, so
it carries no conditional information — yet it occupied ~220 of ~240 input tokens,
made worse because BanglaT5's vocabulary (89.2% Bangla-script) fragments English
identifiers heavily. Removing it cut the median input from 237 to 21 tokens and sped
up training roughly 3×. It is restored by setting `"include_schema": true`.

**Model — two evaluation-side changes.**

- **Checkpoint selection moved to dev execution accuracy** (was exact match), because
  exact match rejects semantically identical SQL.
- **Execution-guided decoding** (Wang et al., 2018): decode 4 beams, return the
  highest-scoring one that *executes without error*, falling back to the top beam.
  `evaluate.py` reports plain top-1 and execution-guided side by side.

Each checkpoint now also saves `banglasql_config.json` recording its exact input
format, so a model can never be queried with a prompt it wasn't trained on. LR 3e-4,
25 epochs, patience 5, 6-epoch warm-up, greedy decoding during training evaluations.

### What it produced

Best epoch 7 (dev exec. 0.5587), early-stopped at 12.

| metric | top-1 beam | execution-guided |
|---|---|---|
| Execution accuracy | 38.6% | **45.5%** |
| Exact match | 21.6% | 23.5% |
| Validity rate | 70.8% | 81.8% |

Against run 1: execution accuracy **27.2% → 45.5%**, validity **63.3% → 81.8%**.
Execution-guided decoding alone was worth **+6.8 points** for no extra training.
Component accuracy: tables 78.4%, group_by 76.6%, aggregate 68.0%, where 59.1%,
join 58.7%, select 40.5%, order_by 35.0%, having 0%.

### Where it failed, and why

**1. Nine query types scored exactly 0%** — `limit`, `order_by`, `order_by_limit`,
`select_column`, `select_distinct`, `join`, `multi_join`, `aggregate_avg`,
`aggregate_max`.

**Why:** every one of them had only **1–2 training templates**. The correlation across
the whole test set is unambiguous:

| training templates for the type | test examples | mean exec. accuracy |
|---|---|---|
| ≤ 2 | 102 (39%) | **31.4%** |
| > 2 | 162 | **54.3%** |

Value-slot augmentation multiplies *within* a template; it cannot teach a query shape
that is nearly absent. The 165 templates were badly unbalanced — `select_where` alone
had 43 while `limit` had 2. Asked "যেকোনো ১০ জন শিক্ষকের নাম", the model produced a
query with no `LIMIT` at all, having seen one `LIMIT` template in training.

**2. Training destabilised at epoch 10** — dev loss jumped 0.22 → 0.80 and execution
accuracy collapsed 0.54 → 0.07 before recovering. LR 3e-4 was too high for the final
phase; the dip also risks early stopping firing on a spurious low.

**3. `select` component accuracy was the weakest at 40.5%** — column projection was
the single most common source of not-quite-right queries.

---

## Run 3 — fix the coverage imbalance

> **Addresses run 2's failures:** 63 new templates concentrated in the starved query
> types (1); learning rate lowered to 2e-4 (2); new templates deliberately widen column
> projection variety (3).

### What it had

**Dataset — 63 new base templates (165 → 228)**, written in standard Cholito Bangla and
concentrated entirely in the thin tail: `limit`, `order_by`, `order_by_limit`,
`select_column`, `select_distinct`, `join`, `multi_join`, `multi_join_where`,
`multi_join_where_order`, `aggregate_avg/max/min/sum/count`, `aggregate_group`,
`aggregate_having`, `join_where_order`, `select_all`.

Every candidate was machine-validated before being accepted: it must execute, return
at least one row, and duplicate neither an existing SQL query nor an existing question.
12 drafts were rejected as duplicates and replaced. New templates also project a wider
set of columns (`course_code`, `email`, `phone`, `building`, `semester`) to attack the
weak `select` score.

Result: **every query type appearing in test now has ≥3 training templates.** The six
remaining single-template types go wholly to train, so they never reach test.

| | train | dev | test |
|---|---|---|---|
| examples | 1382 | 318 | 297 |
| templates | 160 | 34 | 34 |
| distinct SQL queries | **301** (was 224) | 72 | 65 |

**Model.** Unchanged, except **LR 3e-4 → 2e-4** to stop the epoch-10 blow-up.

### What it produced

Best epoch 12 (dev exec. 0.5440), early-stopped at 17. Training wall time ~53 min.
**No instability spike** — the epoch-10 collapse of run 2 (0.54 → 0.07) did not recur;
the equivalent dip was 0.53 → 0.39 and recovered immediately.

| metric | top-1 beam | execution-guided |
|---|---|---|
| Execution accuracy | 40.4% | **41.8%** |
| Exact match | 33.7% | 34.0% |
| Validity rate | 75.4% | 80.8% |

**This 41.8% is not directly comparable to run 2's 45.5%.** The test set was rebuilt
and is weighted toward exactly the query types run 2 failed outright. The fair read is
the component scores and exact match, and those improved substantially:

| | run 2 | run 3 |
|---|---|---|
| Exact match | 23.5% | **34.0%** |
| SELECT component | 40.5% | **68.3%** |
| JOIN component | 58.7% | **76.4%** |
| Aggregate component | 68.0% | **80.5%** |
| `wrong_table` failures | 25 | **6** |

The targeted coverage worked where it was aimed: `limit` 0% → 100%, `select_column`
0% → 100%, `multi_join_where` 13% → 63%, `join` 0% → 50%, `aggregate_group_order`
75% → 100%.

### Where it failed, and why

**1. A flaw in the experimental setup, not the model: the split was unstable.**
`build_dataset.py` *shuffled* template→split assignment every time `templates.json`
changed. So `join_where_order` appeared to collapse 100% → 0%, but its test set had
become a single *different* template (`medium_132`). Same for `aggregate_min` (83% → 0%)
and `multi_join_aggregate` (83% → 0%). With 1–6 templates per query type, per-type
percentages were partly coin flips and run-to-run comparison was not trustworthy.

**2. `wrong_filter` — 53 failures (17.8%).** The comparison operator is wrong:

| gold | predicted |
|---|---|
| `WHERE cgpa < 2.5` | `WHERE cgpa > 2.5` |
| `WHERE cgpa > 3.5` | `WHERE cgpa >= 3.5` |

**3. `wrong_order_by` — 39 failures (13.1%), of which 12 are pure `ASC`/`DESC` flips.**

**Why 2 and 3 share one cause:** value-slot augmentation varies the *number* but never
the *operator* or the *sort direction*. Those stayed welded to their template, so the
model never had to actually read বেশি/কম or আরোহী/অবরোহী — it could infer them from the
rest of the sentence.

**4. A rare word was learned poorly.** বর্ণানুক্রমে ("alphabetically"), introduced in
only 5 of the new templates, was answered with `ORDER BY designation` instead of
`ORDER BY first_name`.

---

## Run 4 — teach polarity, and freeze the split

> **Addresses run 3's failures:** split assignment made deterministic and stable (1);
> polarity augmentation added for comparison operators and sort direction (2, 3);
> বর্ণানুক্রমে tied to the common phrasing via the synonym table (4).

### What it had

**Split made stable across dataset edits.** Assignment is now by a template's
**position** within its query type, using a fixed repeating cycle
(`train, train, test, train, dev, train, train` ≈ 5/7 · 1/7 · 1/7). Template ids are
append-only and zero-padded, so a newly written template always sorts last within its
type and **every existing template keeps its split**. Verified directly: adding a new
template relocates 0 existing templates. The first two cycle positions are train, so a
query type with only one or two templates is never held out and left unlearnable.

This was the last change that moves the test set. From run 5 on, headline numbers are
comparable between runs.

**Polarity augmentation (44 variants).** The operator or sort direction is flipped in
the question and the SQL together:

- বেশি ↔ কম with `>` ↔ `<`
- অবরোহী ↔ আরোহী with `DESC` ↔ `ASC`
- "top N" superlatives — সবচেয়ে বেশি ↔ সবচেয়ে কম, সর্বোচ্চ ↔ সর্বনিম্ন, সবচেয়ে নতুন ↔
  সবচেয়ে পুরনো — with the `ORDER BY` direction

A swap fires only when the operator occurs exactly once in the SQL and its Bangla
marker exactly once in the question, and the result must execute and return rows.
Queries whose direction is carried by `MAX()`/`MIN()` (7 templates) are excluded,
because these rules do not rewrite the aggregate and the flip would be wrong.

Coverage achieved on the training split: 12 of 16 templates that use a comparison now
appear with both `>` and `<`; 21 of 33 that sort appear with both `ASC` and `DESC`. The
remaining gaps were inspected individually and are not fixable by this method — 8
templates state no direction in the question ("নাম অনুযায়ী সাজাও"), 3 already have
their opposite as a separate template, and 1 is the `MAX()` case.

**Synonym fix.** বর্ণানুক্রমে is now mapped to "নাম অনুযায়ী আরোহী ক্রমে", tying the rare
word to the frequent phrasing instead of leaving it to be learned in isolation.

| | train | dev | test |
|---|---|---|---|
| examples | 1521 | 261 | 339 |
| templates | 165 | 29 | 34 |
| distinct SQL queries | **342** | 58 | 79 |

All 28 query types present in train; 0 test SQL queries seen during training.

**Model.** Unchanged from run 3 (question-only input, LR 2e-4, 25 epochs, selection on
dev execution accuracy, execution-guided decoding at evaluation).

### What it produced

Best epoch 6 (dev exec. 0.5172), early-stopped at 11. Training wall time ~24 min. No
instability (lowest dev execution accuracy after epoch 5 was 0.402).

| metric | top-1 beam | execution-guided |
|---|---|---|
| Execution accuracy | 35.1% | **38.0%** |
| Exact match | 19.8% | 21.8% |
| Validity rate | 72.0% | 83.2% |

**The two targeted errors improved decisively:**

| | run 3 | run 4 |
|---|---|---|
| `wrong_order_by` failures | 39 | **3** |
| `wrong_filter` failures | 53 | **42** |
| `order_by` component | 23.5% | **41.8%** |
| `where` component | 54.3% | **65.1%** |
| `having` component | 18.2% | **40.0%** |
| validity rate | 80.8% | **83.2%** |

Polarity augmentation did what it was built for. Seven query types scored 100%:
`limit`, `order_by_limit`, `aggregate_count/max/min/sum`, `select_all`.

**The headline execution accuracy fell 41.8% → 38.0%, but this is not a regression
measurement.** The split-rule change replaced 30 of 34 test templates, so runs 3 and 4
were scored on almost entirely different queries. On the 4 templates common to both:

| | run 3 | run 4 |
|---|---|---|
| execution accuracy | 41.2% | **48.1%** |
| exact match | 27.5% | **44.4%** |

Run 4 is better there, but 4 templates is far too small a sample to settle the question.
It is genuinely unresolved until run 5, which will be the first like-for-like comparison.

### Where it failed, and why

**1. Column projection is now the dominant error — 64 `wrong_select_columns`, and the
`select` component fell to 38.4%.** Most of that jump is a **labelling artifact, not 64
new mistakes**: `categorize_failure` assigns one label in priority order (tables → join
→ aggregate → where → group_by → order_by → select), so fixing `ORDER BY` lets failures
fall through to `select`. All 64 have `ORDER BY` agreeing with gold. Test composition is
almost unchanged (average gold columns 2.26 → 2.14; queries with ≥3 columns 42% → 40%).

Breaking the 64 down:

| pattern | count |
|---|---|
| predicted a subset of gold columns (one missing) | 27 |
| **duplicated a column** (`SELECT course_name, course_name`) | 15 |
| predicted a superset (one extra) | 10 |
| same columns, alias/qualifier differs | 6 |
| genuinely different columns | 6 |

The 15 duplications are a real generation defect. Some of the subset/superset cases
trace to **under-specified gold queries** rather than model error — e.g. "প্রতিটি
শিক্ষার্থী কোন কোন কোর্সে নথিভুক্ত আছে তার নাম সহ জানাও।" has a gold query that also
returns `e.grade`, which the question never asks for; the model's answer without it is
arguably the better one.

**2. Eight query types still score 0%** — `join`, `multi_join`, `multi_join_where_order`,
`multi_join_aggregate`, `aggregate_group`, `aggregate_avg`, `order_by`, `select_distinct`
— now on their new test templates.

**3. The failure-category scheme itself is a limitation.** One label per failure in a
fixed priority order means category counts shift when an earlier-checked component
improves, which makes them hard to read across runs. Reporting per-component accuracy
alongside them (already done) is the reliable view.

### What run 5 should address

1. The duplicated-column defect (15 cases) — the clearest single fix available.
2. Audit gold queries whose question does not state which columns to return; some
   "errors" are dataset ambiguity, and fixing the gold is worth more than fixing the model.
3. Run 5 is the first run with an unchanged test set, so its headline number can finally
   be compared to run 4's directly.

---

## Run 5 — remove the decoding artifact, fix the ambiguous gold

> **Addresses run 4's failures:** duplicated SELECT columns rejected during
> execution-guided decoding (1); the five under-specified gold queries rewritten so the
> question names every column it returns (1).

### What it had

**Same test set as run 4 — verified.** Only five *questions* were reworded; no template
id, query type or SQL changed, and split assignment is position-based on the id. Both
runs' archived datasets contain exactly the same templates in all three splits
(165 / 29 / 34, 0 differences), same 2,121 pairs, 1521 / 261 / 339, 342 distinct train
SQL, 0 leakage, identical model config. **This is the project's first clean A/B.**

**Duplicate-column rejection in execution-guided decoding.** `SELECT course_name,
course_name FROM courses` executes without error, so execution alone cannot filter it
out — it returns an extra column and fails the result comparison. Beam preference is now
(1) executes and has no repeated SELECT column, (2) merely executes, (3) top beam. Safe
by construction: 0 of 228 gold queries select a column twice. The same rule is applied
in the Streamlit demo.

**Five under-specified gold queries rewritten** so the question names every column the
SQL returns (`easy_026`, `easy_073`, `medium_032`, `medium_079`, `medium_133`).

**Model.** Unchanged config from run 4.

### What it produced

| metric | run 4 | run 5 | delta |
|---|---|---|---|
| **Execution accuracy** | 38.0% | **58.7%** | **+20.6** |
| Exact match | 21.8% | 34.8% | +13.0 |
| Validity rate | 83.2% | 91.7% | +8.6 |
| medium difficulty | 34.2% | 61.3% | +27.1 |
| easy difficulty | 45.6% | 53.5% | +7.9 |

Components: group_by +17.6, join +11.6, select +11.5, aggregate +5.4, tables +5.3,
where +4.5. Failures: `malformed_sql` 57 → 28, `wrong_select_columns` 64 → 19,
`wrong_filter` 42 → 33. Zero-scoring query types 8 → 6.

**The duplicate-column rule worked as designed:** predictions containing a repeated
SELECT column fell from **18 to 1**.

### The result is confounded — read it carefully

Despite the identical test set, **run 5 is not a clean test of the two changes**,
because it also trained far longer:

| | run 4 | run 5 |
|---|---|---|
| epochs completed | 11 | **21** |
| best epoch | 6 | **16** |
| best dev execution accuracy | 0.5172 | **0.6054** |

Both runs used the same setting (`patience=5`). Run 4's dev curve plateaued after
epoch 6 — `0.52 0.44 0.44 0.44 0.45 0.40` — and early stopping fired at epoch 11. Run
5's curve happened to improve at epochs 10, 13 and 16, resetting patience each time and
reaching 0.61.

So run 4 was **cut off roughly ten epochs before its peak**. The duplicate-column fix
can account for at most ~5 points (17 recovered predictions out of 339); most of the
remaining ~15 points comes from run 5 simply finding a much better checkpoint. That is
a finding in its own right, and it is the largest single lever discovered so far.

### Where it failed, and why

**1. Early stopping is too aggressive for a noisy dev metric.** Dev is 261 examples from
29 templates, and dev execution accuracy swings by ±0.10 between adjacent epochs. With
`patience=5`, a normal dip ends the run. This cost run 4 about 20 points of test
accuracy purely as a stopping artifact.

**2. One of my own run-5 rewrites made the Bangla ambiguous.** `easy_026` became
"কোর্সগুলোর কোড ও নাম **ক্রেডিট অনুযায়ী** অবরোহী ক্রমে সাজিয়ে দেখাও।" — "নাম ক্রেডিট
অনুযায়ী" reads as a run-on, and the model swaps the sort key with a projection column:

| gold | predicted |
|---|---|
| `SELECT course_code, course_name, credits FROM courses ORDER BY credits DESC` | `SELECT course_code, credits FROM courses ORDER BY course_name DESC` |

All 9 of this template's test failures are this pattern, and it accounts for 9 of the 15
`wrong_order_by` cases. `medium_133` (in train) carries the same run-on. This is a
regression I introduced, not a model failure.

**3. New defect class: `ASC`/`DESC` with no `ORDER BY`** — 5 predictions, all wrong,
e.g. `SELECT first_name, last_name, cgpa FROM students ASC LIMIT 3`. SQLite parses
`students ASC` as a table alias, so the query **executes** and slips past
execution-guided decoding — exactly the same blind spot as the duplicate-column case.
0 of 228 gold queries have `ASC`/`DESC` without `ORDER BY`.

**4. `wrong_aggregate` rose 7 → 17**, of which 14 come from one template (`medium_047`
and its department variants): "...কতটি কোর্সে নথিভুক্ত আছে তার সংখ্যা দেখাও" has a gold
`COUNT(...)` but the model answers `SELECT DISTINCT enrollment_id` — reading "কতটি" as
*list them* rather than *count them*.

**5. Six query types still score 0%**: `join`, `multi_join`, `multi_join_where_order`,
`multi_join_aggregate`, `order_by`, `select_column` (down from 8 in run 4).

### What run 6 should address

1. **Loosen early stopping** (patience 5 → 10). Best-evidenced change available; run 4
   shows the cost of stopping early is far larger than the cost of a few extra epochs.
2. **Reject `ASC`/`DESC` without `ORDER BY`** in execution-guided decoding — same family
   as the duplicate-column rule, provably safe against the gold set.
3. **Repair the `easy_026` / `medium_133` run-on phrasing** by moving the sort clause to
   the front of the sentence.
4. Still open: the six 0% query types, and the COUNT-vs-DISTINCT confusion in
   `medium_047`.

---

## Run 6 — stop stopping early, and repair my own phrasing

> **Addresses run 5's failures:** early-stopping patience raised (1); run-on phrasing
> repaired across all five affected templates (2); orphan `ASC`/`DESC` added to the
> decoding-artifact filter (3).

### What it has

**Same test set as runs 4 and 5 — verified.** Only Bangla questions changed; ids, query
types and SQL are untouched, so split assignment is unaffected. A rebuild confirms 0
template differences in all three splits, and every dataset count is unchanged: 2,121
pairs, 1521 / 261 / 339, 207 value-slot + 44 polarity variants, 342 distinct train SQL,
0 leakage. **Run 6 remains directly comparable to runs 4 and 5.**

**1. Early-stopping patience 5 → 10.** The single best-evidenced change available. Dev
is 261 examples from 29 templates and dev execution accuracy swings ~0.10 between
adjacent epochs, so an ordinary dip looks like a plateau. At `patience=5`:

| | run 4 | run 5 |
|---|---|---|
| epochs completed | 11 | 21 |
| best epoch | 6 | 16 |
| best dev execution accuracy | 0.517 | 0.605 |

Same setting, different luck in the curve — and roughly 20 points of test accuracy
hung on it. Stopping early costs far more here than a few extra epochs do. Max epochs
stays at 25; run 5 peaked at 16 and was declining by 20–21, so the ceiling is adequate.

**2. Orphan `ASC`/`DESC` added to the decoding-artifact filter.** Run 5 produced 5
predictions like `SELECT first_name, last_name, cgpa FROM students ASC LIMIT 3` — all
wrong. SQLite reads `students ASC` as a table alias, so the query **executes** and
passes execution-guided decoding while silently returning unordered rows. This is the
same blind spot as the duplicate-column case, so both now live behind one check,
`is_decoding_artifact()`, used by `evaluate.py` and the demo. Safe against the gold set:
0 of 228 gold queries have `ASC`/`DESC` without `ORDER BY`, and 0 select a column twice.

**3. Run-on phrasing repaired — five templates.** In "A+ পাওয়া শিক্ষার্থীদের নাম ও
কোর্সের নাম **গ্রেড পয়েন্ট অনুযায়ী** অবরোহী ক্রমে দাও।" the projection list runs
straight into the sort key with no clause boundary, and the model merges them. A
systematic scan for questions containing both " ও " and a later "অনুযায়ী" found five:

| template | split | run 5 result | repair |
|---|---|---|---|
| `easy_026` | test | 0/9 | sort clause moved to the front |
| `medium_110` | test | **0/18** | sort clause moved to the front |
| `medium_133` | train | — | rephrased to বর্ণানুক্রমে |
| `medium_113` | train | — | sort clause moved to the front |
| `easy_087` | train | — | sort clause moved to the front |

`easy_026` and `medium_110` alone are 27 test examples — 8% of the test set — that were
scoring 0. `medium_110`'s failure is diagnostic: gold `WHERE e.grade = 'B' ORDER BY
e.grade_point DESC`, predicted `WHERE e.grade_point = 'B'` with no `ORDER BY` — the
model fused `grade` and `grade_point` exactly where the run-on joins them. Two of these
(`easy_026`, `medium_133`) were phrasing I introduced in run 5; the other three date
from run 3.

All rewrites put the sort clause first and keep the verb sentence-final, which preserves
the synonym and polarity augmentation (variant counts are unchanged at 207 / 44).

**Model.** Otherwise unchanged (question-only input, LR 2e-4, 25 max epochs, selection
on dev execution accuracy).

### What it produced

| | run 5 | run 6 |
|---|---|---|
| test execution accuracy | **58.7%** | 54.6% |
| exact match | 34.8% | 34.8% |
| validity rate | 91.7% | 90.9% |
| top-1 exec. acc. (no beam rescue) | 54.6% | 51.0% |
| easy / medium | 53.5% / 61.3% | 64.9% / 49.3% |
| epochs completed | 21 (early stop) | **25 (full)** |
| best epoch / best dev | 16 / 0.605 | 16 / 0.536 |

**Component accuracy**

| component | run 5 | run 6 | |
|---|---|---|---|
| `order_by` | 21.3% | **57.3%** | +36.0 |
| `aggregate` | 71.2% | **86.1%** | +14.9 |
| `having` | 40.0% | 40.0% | — |
| `where` | 69.7% | 60.1% | −9.6 |
| `tables` | 87.0% | 80.8% | −6.2 |
| `select` | 49.9% | 44.5% | −5.4 |
| `join` | 77.9% | 62.0% | −15.9 |
| `group_by` | 70.6% | 43.4% | −27.2 |

`wrong_order_by` disappeared entirely from the failure categories (15 → 0). Orphan
`ASC`/`DESC` in the *chosen* query fell 5 → 1 — the model still emits 15 of them in its
top beam, so the filter is doing real work.

### Where it failed, and why

**1. The three targeted fixes all worked. The headline still fell, because of variance.**

- `order_by` +36 points, `order_by_limit` 0.222 → 1.000, `wrong_order_by` 15 → 0.
- `easy_026` no longer swaps the sort key: run 5 wrote `ORDER BY course_name DESC`,
  run 6 writes `ORDER BY credits DESC`. The run-on repair did exactly what it was for.
- Training ran the full 25 epochs and never triggered early stopping, so patience is no
  longer the binding constraint.

Yet the headline moved −4.1 points. Grouping predictions by base template explains why:
the test set has **34 templates**, and each is effectively all-or-nothing. Two templates
account for the entire drop — `medium_096` 15/15 → 4/15 and `medium_035` 9/9 → 0/9, a
loss of 20 examples, or 5.9 points. A cluster bootstrap over templates puts run 5 at
0.404–0.762 and run 6 at 0.373–0.711. **The two runs are statistically
indistinguishable, and so were several earlier pairs.** The entire dev curve was worse
in run 6 from epoch 1 onwards (epoch 4: 0.364 vs 0.517) despite `SEED = 42` and only
30 of 1,521 training questions changed — this is GPU nondeterminism, not a data effect.

**2. The gold SELECT list asks for columns the question never names — my own residue.**
`easy_026` and `medium_110` are still 0/9 and 0/18, but for a *different* reason now:

```
Q    ক্রেডিট অনুযায়ী অবরোহী ক্রমে সাজিয়ে কোর্সগুলোর কোড ও নাম দেখাও।   ("code and name")
GOLD SELECT course_code, course_name, credits FROM courses ORDER BY credits DESC;
R6   SELECT course_code, course_name            FROM courses ORDER BY credits DESC;
```

The model produced exactly what the question asked for. The gold wants a third column
(`credits`; `e.grade_point` in `medium_110`) that neither the run-5 nor the run-6
question mentions. I fixed the run-on and left the under-specification in place.

This is systematic, not two stray cases: across all 228 templates the `ORDER BY` column
appears in the `SELECT` list **28 times and is absent 7 times**, with nothing in the
Bangla to distinguish them. 18 of run 6's 154 failures are queries where every clause
matches gold and only the `SELECT` list differs in length. `select` is the
worst-performing component in every run (44.5%) and the only one that has never
improved.

**3. Aggregate aliases are arbitrary, and `HAVING` depends on them.** `medium_035`
(9 examples, `aggregate_having`) fell from 9/9 to 0/9:

```
GOLD ... COUNT(e.enrollment_id) AS count           ... HAVING count > 5;
R6t1 ... COUNT(e.enrollment_id) AS enrollment_count ... HAVING student_count > 5;   ← alias undefined → fails to run
R6   ... COUNT(e.enrollment_id) AS enrollment_count ... WHERE e.grade > 5;          ← beam rescue picked a worse query that runs
```

The gold corpus uses **36 distinct alias names** for roughly a dozen concepts — a single
`COUNT` is variously `count`, `student_count`, `course_count`, `enrollment_count`,
`total_students`, `total_records`. The alias is not recoverable from the Bangla, so the
model guesses, and when one of the **12 `HAVING` templates** guesses inconsistently
between `SELECT` and `HAVING`, the query does not execute. Execution-guided decoding
then makes it worse by falling through to a query that runs but is semantically wrong.
Alias choice also suppresses exact match everywhere, not just here.

**4. Twelve templates still score 0** — 117 of 339 examples (34.5%), unchanged in count
from run 5 though not the same twelve: `medium_110`, `easy_015`, `medium_017`,
`medium_089`, `medium_035`, `easy_026`, `easy_053`, `medium_032`, `easy_035`,
`medium_109`, `medium_138`, `medium_005`. With 34 templates in the test set, moving any
three of these is worth ~8 points — which is the same magnitude as the noise.

---

## Run 7 — repair the data, not the model

> **Addresses run 6's finding:** the pipeline was sound but the measurement was not.
> Run 6's three fixes all landed and the headline still fell, because the gold data
> contained errors, unanswerable questions and clauses that excluded nothing.

**The test set changes here.** Gold SQL and the database both moved, so run 7's numbers
are not comparable to runs 4–6. That break is deliberate and taken in one pass rather
than spread over several runs. Given run 6's 95% CI of 0.37–0.71 there was little
comparability left to protect.

### What it has

**An audit of all 228 templates.** Every question was read against its SQL and every
gold query executed. 13 were wrong, 10 internally inconsistent, and one ambiguity
affected 35.

**1. Four questions had no answer.** `easy_035`, `easy_043` (both test), `medium_034`
and `medium_042` returned 0 rows — 12 test examples, 3.5%, whose correct answer was an
empty table. Rather than weaken the questions, the database was repopulated so the facts
they ask about exist:

| change | why |
|---|---|
| per-department CGPA means (2.88–3.70) | `AVG(cgpa) > 3.5` had no answer; every department sat near 3.0 |
| fixed session dates, two named by the templates | date filters landed on whatever the random draw produced |
| attendance drawn from the student's own enrolments | a student could have attendance for a course they never took |
| a fifth of students attend poorly | nobody had more than 1–2 absences, so "absent more than 5 times" was unanswerable |
| enrolments driven per student, 2–6 courses each | no student or course is empty; 500 → ~1,030 enrolments, 500 → ~4,200 attendance rows |
| grades scatter around the student's CGPA, plus a 5% failure rate | grades were uncorrelated with CGPA, so a 3.9 student was as likely to fail as a 2.1 student |

**2. Nine question/SQL disagreements.** `easy_028` asked for the enrolment ID and
returned `student_id`. `medium_047` and `medium_094` answered "how many courses" with
`COUNT(enrollment_id)` — `medium_047` had scored 0/15 for three runs because the *gold*
was wrong, not the model. `medium_045` counted attendance rows for "how many people".
`medium_019` sorted when nothing asked it to. `medium_056` returned a column the
question never named. `medium_022` answered "which courses" with a bare `course_id`.
Four templates had `tables_used` metadata that disagreed with their own SQL.

**3. Clauses that excluded nothing.** A `HAVING` that keeps every group is decoration:
a prediction that drops it scores exactly as well as the gold. `medium_035`'s
"more than 5 students" selected all 80 courses, `medium_134`'s "more than 5 courses" all
7 semesters. Course popularity and semester size are now uneven, so thresholds bite.
Two `ORDER BY` clauses could not reorder anything — `medium_110` sorted A+ rows by grade
point, and every A+ row has grade point 4.0. A scan over all five clause types
(`WHERE`, `HAVING`, `DISTINCT`, `ORDER BY`, `LIMIT`) now reports **0** that change
nothing.

**4. Answers shared across splits.** Six pairs of templates returned identical rows
through different SQL — "grade point above 3.5" and "A or A+" select the same enrolments,
because the grading scale puts A at 3.75 and A− at 3.50. Where one sat in train and its
twin in test, the test question could be answered by reciting the training query. Three
pairs were broken by changing thresholds and by removing the forced 4.00 CGPA that made
the highest CGPA and the highest grade point the same number. For the rest,
`build_dataset.py` now drops any dev/test example whose gold **result set** already
appears in train — textual SQL dedup cannot see this. It removed 24 examples.

**5. Two conventions the model could only guess at.**
- *Aggregate aliases:* 36 names for about a dozen concepts; `COUNT` was variously
  `count`, `student_count`, `total_students`, `total_records`. The alias is not
  recoverable from the Bangla, and 12 templates reference it in `HAVING`, where an
  inconsistent guess produces a query that will not execute — that is the whole of
  `medium_035`'s 9/9 → 0/9 swing in run 6. Now deterministic: `COUNT` → `<noun>_count`
  (or `<status>_days`), `SUM` → `total_<column>`, `AVG` → `avg_<column>`,
  `MAX`/`MIN` → `max_/min_<column>`.
- *Join type:* the same "প্রতিটি বিভাগে কতজন" frame was `LEFT JOIN` in five templates and
  `JOIN` in four. Only one `LEFT JOIN` changed its result, so all seven are now plain
  `JOIN` and the join type is no longer a coin flip.

**6. Every returned column is now named in the question.** The `ORDER BY` column appeared
in the `SELECT` list in 28 templates and was absent in 7, with nothing in the Bangla to
tell them apart — which is why `select` was the worst component in every run (44.5%) and
the only one that never improved. A column-to-Bangla cue scan now reports **0** returned
columns with no cue in the question.

**Model.** Unchanged from run 6.

### Verification

| check | result |
|---|---|
| templates that execute and return rows | 228 / 228 |
| clauses that exclude or reorder nothing | 0 |
| templates returning identical rows to another | 0 |
| returned columns not named in the question | 0 |
| `HAVING` aliases undefined in `SELECT` | 0 |
| `tables_used` disagreeing with the SQL | 0 |
| augmented examples with an empty or failing gold | 0 / 2,097 |
| answer sets shared between train and dev/test | 0 |
| database and splits byte-identical across rebuilds | yes |

Dataset: 2,097 pairs — 1,506 train / 240 dev / 351 test, from 165 / 28 / 34 templates,
337 distinct train SQL, 0 SQL leakage, all 28 query types present in train.

### What it produced

Every prediction the audit made about the *components* was confirmed. The headline did
not move.

| | run 6 | run 7 |
|---|---|---|
| test execution accuracy | 54.6% | 55.0% |
| **test exact match** | 34.8% | **50.4%** |
| validity rate | 90.9% | 85.2% |
| best dev execution accuracy | 0.536 (ep 16) | **0.629** (ep 17) |
| best dev exact match | 0.399 | **0.583** |

| component | run 6 | run 7 | |
|---|---|---|---|
| `select` | 44.5% | **75.8%** | **+31.3** |
| `having` | 40.0% | **100%** | +60.0 |
| `group_by` | 43.4% | 60.9% | +17.5 |
| `join` | 62.0% | 68.3% | +6.3 |
| `where` | 60.1% | 65.3% | +5.2 |
| `tables` | 80.8% | 85.8% | +5.0 |
| `order_by` | 57.3% | 53.9% | −3.4 |
| `aggregate` | 86.1% | 78.6% | −7.5 |

`select` had been 44–50% in every run and had never improved; naming every returned
column in the question moved it 31 points. `aggregate_having` went 0.400 → **1.000** once
aliases became deterministic. Exact match rose 15.6 points, the largest jump in the
project, which is what alias standardisation predicts: the same query written one way
every time.

**Test-set note.** 351 examples against run 6's 339, and the gold changed, so the
headline is not a like-for-like comparison. The cluster bootstrap over the 34 test
templates gives run 7 a 95% CI of **0.371–0.727** — as wide as ever. The component table
and exact match are the readable signal here; the headline is not.

### Where it failed, and why

**1. Alias standardisation bought `having` and cost validity.** Malformed SQL rose 31 →
52, and **23 of those 52 reference an alias the query never defines**:

```
Q     প্রতিটি কোর্সের গড় গ্রেড পয়েন্ট হিসাব করো।
GOLD  SELECT c.course_name, ROUND(AVG(e.grade_point), 2) AS avg_grade_point FROM ... GROUP BY c.course_name;
PRED  SELECT c.course_name, avg_grade_point                                 FROM ... GROUP BY c.course_name;
```

The model learned the alias as a reliable token and now emits it *instead of* the
aggregate expression that defines it. Same shape in `ORDER BY course_count` with no
`COUNT` anywhere, and in `JOIN joining_year` — a column hallucinated into a table
position. This is the direct cost of making aliases predictable, and it is worth paying:
`having` went to 100% and exact match rose 15.6 points. But it is now the single largest
failure category.

**2. "পরে" (after) never appears in training.** `easy_015` scored **0/15**, predicting
`joining_year < 2015` for "২০১৫ সালের পরে যোগ দেওয়া" (*joined after 2015*). The cause is
exact and checkable: `POLARITY_RULES` pairs বেশি/কম and আরোহী/অবরোহী but has no
আগে/পরে pair. Every "আগে" (`<`) example sits in train via `easy_013`; every "পরে" (`>`)
example sits in test via `easy_015`. The model has never once seen পরে → `>`. A marker
sweep over the corpus finds this is the only comparison word with that property.

**3. Two templates still return a column the question does not ask for.** `easy_026`
(**0/9**) and `medium_113` name their projections with "ও" and do not include the sort
column, but the gold returns it. The run-7 cue scan passed them because the column's
Bangla word *is* in the question — inside the sort clause ("ক্রেডিট অনুযায়ী"), not the
projection list. The check could not tell "sort by credits" from "show credits".

**4. Two of the audit's own fixes are not yet learnable.** `medium_047` (**0/15**) now
correctly asks for `COUNT(DISTINCT e.course_id)` and the model answers `COUNT(*)` — the
`COUNT(DISTINCT ...)` pattern occurs in only three templates corpus-wide.
`medium_067` (**0/12**), rewritten to a semester filter, has the model folding the `WHERE`
into the `ON` clause: `JOIN enrollments e ON c.semester = 'Fall 2023'`.

**5. `medium_110` drops `ORDER BY` when the sort column is also the filter column.**
Gold is `WHERE e.grade_point > 3.5 ORDER BY e.grade_point DESC`; the prediction is
correct in every respect except the missing `ORDER BY`. 21 examples.

**6. 36 training targets are silently truncated.** `max_target_length` is 128 and 36
train plus 3 dev gold queries tokenize longer, up to 139 — `medium_022`, which the audit
gave a fourth `JOIN`. No test example is affected, so this costs training signal rather
than score.

**Still at zero:** 13 templates, 138 of 351 examples (39%) — `easy_015`, `easy_026`,
`easy_035`, `easy_053`, `medium_005`, `medium_015`, `medium_017`, `medium_032`,
`medium_047`, `medium_067`, `medium_089`, `medium_110`, `medium_138`.

---

## Run 8 — pay back run 7's costs *(prepared, not yet trained)*

> **Addresses run 7's failures:** the missing আগে/পরে polarity pair (2), the bare-alias
> failure that alias standardisation created (1), the two questions whose sort column the
> cue scan could not see (3), and the silently truncated training targets (6).

**The test set is unchanged where it counts.** Aliases affect column *headers*, not
values, so every gold result set is byte-identical to run 7's. Execution accuracy stays
comparable; exact match does not, since the gold SQL text changed.

### What it has

**1. আগে/পরে added to `POLARITY_RULES`.** The clearest finding of run 7 and a one-line
fix. Every "আগে" (before, `<`) example lived in train via `easy_013` and every "পরে"
(after, `>`) example in test via `easy_015`, so the model had never seen পরে mean `>` and
scored **0/15**. The rule now generates the counterpart, and "পরে" → `>` appears in train,
dev and test. A marker sweep confirms no comparison word is left with a polarity absent
from training.

**2. Every aggregate alias removed — 81 templates.** Run 7 made aliases deterministic,
which took `having` from 40% to 100% and lifted exact match 15.6 points. It also taught
the model that `avg_grade_point` is a reliable token, and it began emitting the alias
*instead of* the expression defining it — 23 of run 7's 52 malformed queries. Standard SQL
does not need the alias; `HAVING` and `ORDER BY` take the aggregate directly:

```sql
-- run 7
SELECT c.course_name, COUNT(e.enrollment_id) AS student_count FROM ... HAVING student_count > 5;
-- run 8
SELECT c.course_name, COUNT(e.enrollment_id)                   FROM ... HAVING COUNT(e.enrollment_id) > 5;
```

There is now exactly one way to write each aggregate, so the failure cannot be expressed
rather than merely being rarer. The consistency burden that broke `medium_035` in run 6 —
choosing one name in `SELECT` and a different one in `HAVING` — disappears too, because
the second occurrence is a copy of the first rather than a name to invent.

**3. Two questions now name the sort column they return.** Run 7's cue scan asked whether
a column's Bangla word appeared in the question at all. "ক্রেডিট অনুযায়ী ... কোড ও নাম
দেখাও" passed, because ক্রেডিট is present — inside the sort clause, not the projection
list. The gold returns credits regardless, so the model produced exactly what was asked
and `easy_026` scored **0/9**. The check now asks whether the word appears anywhere other
than immediately before a sort marker; `easy_026` and `medium_113` both list their
projections with "ও" while excluding the sort column, and both now name it.

The other 14 templates the sharper check flags give no projection list at all
("সকল শিক্ষার্থীকে CGPA-র অবরোহী ক্রমে সাজাও"), where returning the sort column is the
natural reading. They are left alone.

**4. Target length sized off the longest query, not p95.** `preprocess_check.py` rounded
from the 95th percentile (117 tokens → 128), while the longest gold query is 139. Run 7
trained with **36 train and 3 dev targets cut short** — the model was shown incomplete SQL
and told it was correct. Sizing from the maximum gives 160, and 0 examples truncate.

### What is deliberately not changed

- `medium_047` (**0/15**) answers `COUNT(*)` where gold wants `COUNT(DISTINCT e.course_id)`.
  This is not a coverage hole: `COUNT(DISTINCT ...)` appears in three train templates,
  `medium_094` among them with the same "কতটি কোর্সে" phrasing. The signal is present and
  the model has not learned it — a modelling problem, not a data one.
- `medium_110` (**0/21**) drops `ORDER BY` when the sort column is also the filter column.
- `medium_067` (**0/12**) folds the `WHERE` into the `ON` clause.

### Verification

| check | result |
|---|---|
| templates with an alias remaining | 0 / 228 |
| templates that execute and return rows | 228 / 228 |
| gold result sets changed by alias removal | 0 |
| clauses that exclude or reorder nothing | 0 |
| templates returning identical rows to another | 0 |
| returned columns not named in the question | 0 |
| comparison markers with a polarity absent from train | 0 |
| training targets truncated at `max_target_length` | 0 (was 39) |
| answer sets shared between train and dev/test | 0 |
| splits byte-identical across rebuilds | yes |

Dataset: 2,064 pairs — 1,494 train / 243 dev / 327 test, from 165 / 28 / 34 templates,
333 distinct train SQL, 0 leakage. The answer-leakage guard dropped 45 examples, up from
24, because removing aliases made more queries textually identical.

### What to check when it finishes

1. `easy_015`, which was 0/15 purely because পরে never appeared in training. This is the
   cleanest prediction in the project — if it does not move, the polarity augmentation is
   not reaching the model.
2. Validity rate and `malformed_sql`. 23 of run 7's 52 malformed queries referenced an
   undefined alias; that category should be empty.
3. `having`, which run 7 took to 100%. Repeating the aggregate expression is a different
   burden from reusing a name, so this is the main risk the change introduces.
4. `easy_026`, 0/9 across runs 5, 6 and 7 for three different reasons.

---

## Reproducing any run

```bash
python create_database.py    # deterministic (seed 42) — same DB every time
python build_dataset.py      # deterministic — same splits every time
python preprocess_check.py   # tokenizer coverage + sequence-length profiling
python train.py              # GPU required; use colab_train.ipynb
python evaluate.py           # writes logs/test_results.json + test_predictions.json
```

All scripts seed `random`, `numpy` and `torch` with 42. `logs/test_predictions.json`
holds every prediction with its gold query, failure category and execution error,
which is the source for all error analysis above.
