# Prompt Inventory

Every LLM prompt used in the project, reproduced verbatim from the source, with the
file it lives in and the stage that uses it. Three components call an LLM:

1. **Detection / explanation pipeline** (Qwen3 8B via Ollama) — zero-shot, random
   few-shot, RAG few-shot, and the cross-lingual German/Italian runs. All four share
   one system prompt and output schema; only the user message differs.
2. **HyDE retrieval** (Qwen3 8B) — generates a hypothetical snippet used for retrieval.
3. **Synthetic data augmentation** (Qwen3 8B) — generates rare-class training examples.
4. **Prometheus LLM-as-judge** (`prometheus-eval/prometheus-7b-v2.0`) — scores rationale
   quality. (Rewrite quality is measured with BERTScore, not an LLM prompt.)

Source map:

| Prompt | File |
|---|---|
| System prompt + output schema | `src/pipelines/prompts.py`, `src/pipelines/schema.py` |
| Italian signal appendix | `src/pipelines/prompts.py` |
| Zero-shot user message | `src/pipelines/prompts.py` |
| Few-shot user message (random + RAG) | `src/pipelines/prompts.py` |
| HyDE hypothesis generation | `src/retrieval/retrieve.py` |
| Synthetic augmentation (system + builder) | `src/data/augment.py` |
| Prometheus rationale-judge rubric | `src/evaluation/rationale_judge.py` |

---

## 1. Detection / explanation pipeline

### 1.1 System prompt (shared by all pipelines)

`src/pipelines/prompts.py` (`SYSTEM_PROMPT`). Used by `zero_shot.py`, `random_few_shot.py`,
`rag_few_shot.py`, and `cross_lingual_eval.py`. The `{OUTPUT_SCHEMA_STR}` placeholder is
filled with the schema in §1.2.

```text
You are an expert in consumer psychology and e-commerce manipulation.
Your task: analyse a product page text and identify any dark pattern — a deceptive copy or UI practice that manipulates consumers.
The text may be in any language (English, German, Italian, or other). Analyse it in the original language — do not translate.

Respond with ONLY a valid JSON object following this exact schema (no markdown fences, no extra keys):
{OUTPUT_SCHEMA_STR}

Dark pattern categories and their key signals:
  Scarcity       — false or exaggerated claims of limited stock or availability
  Urgency        — artificial time pressure (countdown timers, "today only", "expires soon")
  Social Proof   — manipulated social validation ("1,000 people viewing", fake or inflated reviews)
  Misdirection   — attention diverted from important options (pre-ticked boxes, buried opt-outs, confirmshaming)
  Obstruction    — deliberately hard to cancel, unsubscribe, or opt out; excessive notice periods
  Forced Action  — users must take unwanted steps (accept marketing emails, create an account)
  Sneaking       — hidden charges, auto-added items, undisclosed auto-renewal subscriptions
  Not Dark Pattern — transparent, honest product text with no manipulation; clearly states guest checkout is available, no lock-in, easy cancellation

Rules you must follow:
1. reasoning_steps MUST be populated FIRST — think step by step before committing to a label.
2. evidence_span MUST be an exact verbatim substring copied from the input text.
3. confidence is your certainty [0.0–1.0] for the chosen label.
4. For "Not Dark Pattern" set evidence_span to the most representative neutral phrase.
5. Use "Uncertain" only when you genuinely cannot determine the label.
6. rewrite must rewrite the product text to remove the dark pattern while preserving the core offer.
   - If the text is a dark pattern: produce a complete, natural sentence even if the input is short (e.g. "LAST 1 LEFT" → "This item is available — add it to your cart.").
   - If the label is "Not Dark Pattern": copy the input text unchanged.
```

### 1.2 Output schema (embedded in the system prompt)

`src/pipelines/schema.py` (`OUTPUT_SCHEMA_STR`). This is the exact JSON template inserted
in place of `{OUTPUT_SCHEMA_STR}` above.

```text
{
  "reasoning_steps": [
    "<step 1: identify any manipulation tactics present in the text>",
    "<step 2: match tactics to the most fitting dark pattern category>",
    "<step 3: assess confidence and note any ambiguities>"
  ],
  "label": "<one of: Scarcity | Urgency | Social Proof | Misdirection | Obstruction | Forced Action | Sneaking | Not Dark Pattern | Uncertain>",
  "confidence": <float between 0.0 and 1.0>,
  "psychological_mechanism": "<name of cognitive bias or persuasion principle>",
  "evidence_span": "<verbatim substring from the input text>",
  "harm_dimension": "<short phrase describing consumer harm>",
  "rationale": "<1-2 sentence explanation>",
  "rewrite": "<full product text with dark pattern removed, core offer preserved>"
}
```

### 1.3 Italian signal appendix (cross-lingual Italian runs only)

`src/pipelines/prompts.py` (`ITALIAN_SIGNAL_APPENDIX`). Appended to the system prompt by
`system_prompt_for("it")`; English and German callers get the base prompt unchanged.

```text
Italian-specific surface cues (use these to disambiguate when the input is in Italian):
  Scarcity       — "solo X rimasti", "ultimi X pezzi", "ultime camere", "in esaurimento", "edizione limitata"
  Urgency        — "solo per oggi", "ultima chance", "subito", "scade tra", "termina tra", "affrettati", "ultime ore", "follia del giorno"
  Social Proof   — "prenotato X volte", "scelto da X utenti", "il più venduto", "hanno acquistato", "salvato in X liste dei desideri"
  Misdirection   — "prezzo di pubblicazione", "prezzo consigliato", "prezzo imbattibile", "risparmi rispetto a", strikethrough discount math
  Obstruction    — "raccomandata A/R", "PEC", "Posta Elettronica Certificata", "numero verde (orari limitati)", "area clienti", "modulo PDF", "X giorni di preavviso", "contattare il servizio clienti via telefono", "non è possibile cancellare online"
  Forced Action  — "devi accettare i cookie di profilazione", "devi verificare il numero", "obbligatorio inserire", "iscriviti alla newsletter per", "registrati per continuare la lettura"
  Sneaking       — "applicato in fase di checkout", "costo di servizio", "spese di gestione obbligatorie", "contributo ambientale RAEE", "costi doganali a carico del destinatario", "non include", "rinnovo automatico", auto-added paid subscription
  Not Dark Pattern — "consegna standard 3-5 giorni", "garanzia soddisfatti o rimborsati", "recensioni verificate", "politica di reso 30 giorni", neutral product/category labels, last-access timestamps
```

### 1.4 Zero-shot user message

`src/pipelines/prompts.py` (`format_zero_shot_prompt`). `{text}` is the target snippet.

```text
Analyse the following product text and return a JSON object following the schema.

Product text: """{text}"""

Output JSON:
```

### 1.5 Few-shot user message (random few-shot and RAG)

`src/pipelines/prompts.py` (`format_few_shot_prompt` + `_format_example_block`). The same
template serves both random few-shot and RAG; only the source of the `k` examples differs
(random sampling vs. FAISS retrieval). `{text}` is the target snippet; the example block is
built from the retrieved/sampled examples.

Per-example block (`_format_example_block`), repeated for each example; the `Key phrase`
line is included only when an `evidence_span` is available:

```text
Here are labelled examples to guide your analysis:

--- Example {i} ---
Product text: """{example_text}"""
Label: {category}
Key phrase: "{evidence_span}"
```

Full assembled user message:

```text
{example_block}---

Now analyse the following product text and return a JSON object following the schema.

Product text: """{text}"""

Output JSON:
```

---

## 2. HyDE hypothesis generation

`src/retrieval/retrieve.py` (the `hyde` retrieval strategy). Generated at temperature 0.7;
the hypothesis is encoded and used for KNN retrieval instead of the raw query. On failure
the pipeline falls back to the raw query. `{query}` is the target snippet.

```text
You are an expert in e-commerce dark patterns.
Given the product text below, write ONE short hypothetical product description that exemplifies the most likely dark pattern it contains. Your output must be a realistic product snippet (1-3 sentences), not an explanation.

Product text: """{query}"""

Return JSON: {"hypothesis": "<your hypothetical dark-pattern snippet>"}
```

---

## 3. Synthetic data augmentation

`src/data/augment.py`. Generates rare-class training snippets (Forced Action, Sneaking,
Obstruction) at temperature 0.8, three per call, with product type rotated across calls
for diversity. Output is filtered by a bigram-Jaccard near-duplicate filter (> 0.4).

### 3.1 Augmentation system prompt

`SYSTEM_PROMPT` in `augment.py`:

```text
You are an expert in dark patterns in e-commerce UX and persuasive design. You generate realistic synthetic product-page text snippets that exhibit a specific dark pattern. Output only valid JSON. Do not explain yourself.
```

### 3.2 Augmentation user message (template)

`build_prompt(category, seed_example, product_type)`. `{description}` is the category
description from §3.3; `{product_type}` rotates over the list in §3.4; `EXAMPLES_PER_CALL`
is 3.

```text
Dark pattern category: {category}
Definition: {description}

Product context: {product_type}

Real example of this pattern:
"""{seed_example}"""

Generate {EXAMPLES_PER_CALL} new product-page text snippets that exhibit the same dark pattern. Each snippet should feel authentic, be about a {product_type}, and use different phrasing from the example. Keep each snippet under 60 words.

Return as JSON with this exact structure:
{"examples": ["snippet 1", "snippet 2", "snippet 3"]}
```

### 3.3 Category definitions (inserted as `{description}`)

`CATEGORY_DESCRIPTIONS` in `augment.py`:

```text
Forced Action — the user is required to consent to marketing emails, terms of service, or data sharing as a condition of completing a purchase or registration, with no genuine opt-out.

Sneaking — hidden fees, automatically added items (e.g. insurance, memberships, charity donations), or costs that appear only at the final checkout step, not shown upfront.

Obstruction — cancelling a subscription or membership is made deliberately difficult: buried phone numbers, long hold times, vague instructions, or multi-step processes designed to frustrate the user into giving up.
```

### 3.4 Product types rotated as `{product_type}`

`PRODUCT_TYPES` in `augment.py`:

```text
online clothing retailer, electronics store, hotel booking platform,
streaming subscription service, grocery delivery app, beauty and cosmetics brand,
fitness and gym membership, travel booking website, software as a service (SaaS),
food delivery platform, pet supplies store, home appliances retailer,
book subscription club, meal kit delivery service, online pharmacy,
music streaming platform, gaming subscription service, flower delivery website
```

---

## 4. Prometheus rationale-judge rubric

`src/evaluation/rationale_judge.py`. Judge model `prometheus-eval/prometheus-7b-v2.0`, run
reference-free: the rubric criterion stands in for a gold reference. Each prediction is
scored 1–5 on each of four criteria; the criterion definition is inserted into the template
once per criterion.

### 4.1 Scoring template

`_RUBRIC_TEMPLATE`. `{input_text}`, `{label}`, `{reasoning_steps}`, `{rationale}` come from
the prediction; `{criterion_name}`/`{criterion_description}` come from §4.2.

```text
###Task Description:
An instruction (might include an Input inside it) and a response are given.
Evaluate the quality of the response using the criterion below.
Score the response on a scale of 1 to 5.

###Instruction:
Analyse the following product page text and identify any dark pattern present.
Provide reasoning steps and a rationale explaining your label choice.

Product text: """{input_text}"""

###Response to evaluate:
Label: {label}

Reasoning steps:
{reasoning_steps}

Rationale: {rationale}

###Criterion:
{criterion_name}: {criterion_description}

###Score Rubric:
Score 1: The response completely fails to satisfy this criterion.
Score 2: The response partially satisfies this criterion with significant gaps.
Score 3: The response satisfies this criterion moderately well.
Score 4: The response satisfies this criterion well with only minor issues.
Score 5: The response fully satisfies this criterion without any issues.

###Feedback:
```

### 4.2 The four criteria (`_CRITERIA`)

```text
Coherence: Do the reasoning_steps logically lead to the predicted label? Is the rationale internally consistent with the steps?

Faithfulness: Are all claims in the rationale grounded in the input text? The rationale must not introduce evidence not present in the product text.

Specificity: Does the rationale cite concrete words or phrases from the input text rather than making generic statements about the label category?

Non-Circularity: Does the rationale explain WHY the tactic is manipulative to the consumer, rather than simply restating what the label name means?
```

The score is parsed from Prometheus output via the `[RESULT] N` convention (falling back to
the last standalone digit 1–5). The four criterion scores sum to a composite in [4, 20].

---

## 5. Rewrite quality (no LLM prompt)

`src/evaluation/rewrite_quality.py` evaluates the `rewrite` field with **BERTScore** (STA:
semantic textual alignment between input and rewrite), not an LLM prompt, so there is no
prompt to document here.
