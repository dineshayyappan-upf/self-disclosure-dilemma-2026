# Methods and files

## Scope

Five models, 34 wellbeing questions, 44 disclosure conditions and four phrasings.
`data/responses.jsonl.gz` contains 30,090 final merged cells: 29,920 tailored
responses (including eight empty cells) and 170 no-disclosure controls. The
29,912 nonempty tailored cells enter distance analysis. Empty cells remain in
behavioral denominators and contribute zero words.

The archived 643-response reliability sample also contains auxiliary questions
and an additional model run, as described in the paper. The supplied judgments
are Opus 4.6 and Qwen3, the pair reported in the paper. Saved parsing outcomes
are retained. Inter-model agreement measures automated scoring consistency.

## Distances

Complete answers are encoded with `Alibaba-NLP/gte-modernbert-base`, revision
`e7f32e3c00f91d699e8c43b53106206bcc72bb22`, using the model-defined CLS pooling
and L2 normalization. The maximum input is 8,192 tokens; no input was truncated.
Statistical text removes hidden reasoning, exact query/disclosure echoes and
direct experimental group names, and normalizes links, whitespace and typography.
`code/cleaning.py` contains the recorded rules. Original response text is retained.

Each tailored answer is compared to the archived original no-disclosure answer
for the same model and question. Subtract the equal-pair mean of 170 archived
original/rephrasing distances: **0.05535751899048206**. Negative values remain.
Zero denotes the average rephrasing distance, not identical text or a quality
threshold. Merged no-disclosure controls used for tag tables can differ from
the archived original controls used for embeddings; both are supplied.

Model and model/category means average responses within each question, then
weight the 34 questions equally. Phrasing summaries pool observed responses
across Opus, Qwen3 and GPT-Safe. Intervals are pointwise 95% percentile intervals
from 5,000 whole-question resamples, with the saved seeds and rephrasing mean
held fixed.

## Behavioral results

The six binary tags are recorded Opus judgments. Completion and refusal are
independent and can co-occur. Table rates are pooled Yes counts divided by
record counts. Parenthesized SDs describe variation across the strata named in
the result CSVs; they are not confidence intervals. Condition radars pool 34
questions and four phrasings per condition and model. Word counts use original
visible text, remove hidden reasoning and count contractions as single words.

## Data guide

All compressed text files use gzip and UTF-8. JSONL has one record per line.
Model/query/condition/phrasing identify a response cell; these are experiment
identifiers, not people.

| File | Contents |
| --- | --- |
| `data/responses.jsonl.gz` | Full primary response text, exact query/disclosure, model and condition identifiers, saved six-tag labels and raw judge output. `analysis_text_sha256` pins the cleaned text for each nonempty tailored cell. |
| `data/rephrasing.jsonl.gz` | 170 full original/rephrased no-disclosure pairs, keyed by model/query; hashes pin both cleaned texts. |
| `data/gte/response_scores.csv.gz` | 29,912 saved cosine distances, centered distances and encoder token counts; keys use `model_id`, `question_id`, `condition_id`, `phrasing_id`. |
| `data/gte/rephrasing_pairs.csv` | 170 saved rephrasing distances. |
| `data/gte/summary_centered_distance.csv` | 33 means, intervals and bootstrap seeds. Scope a=model, b=model/category, c=category/phrasing. |
| `data/tables/` | Saved behavioral results, model word lengths, condition rates and the six reliability metrics reported in the paper. |
| `data/reliability_sample.jsonl.gz` | The 643 archived responses, sample IDs and original tags. These texts can differ from final merged cells. |
| `data/reliability_judgments.jsonl.gz` | 1,286 saved Opus/Qwen judgments, including raw outputs and parse flags, joined by `sample_id`. |
| `instruments/` | Exact query/disclosure inventory, rubric, judge and rephrasing prompts, model IDs and collection settings. Categories: baseline=majority, subclinical=non-clinical challenges, type1=protected non-disability, type2=protected disability. |
| `qualitative/` | Codebook, 12 discovery and 12 verification pairs, author notes and source status matrix. |
| `data/source_hashes.json` | Original-input and source-document hashes and the release transformations. |

## Optional fresh encoding

```sh
python3 -m pip install -r requirements-encoding.txt
python3 code/encode.py
```

The encoder is downloaded at the pinned revision unless `--model-path` names
a local copy. Use `--device` and `--dtype` to select hardware and precision;
the archived run used Apple MPS with float16. `--limit 2` is a small encoding
check using only the required pairs and the archived pooled mean. Fresh results and differences from saved distances are written to
`outputs/reencoded/`; saved data are unchanged. Hardware, precision and batching
can introduce numerical differences. A full encoding run is separate from the
offline numerical reproduction.

## Optional generation, judging and rephrasing

`code/collect.py` constructs prompts from the released instruments and uses the
recorded request adapter. Without `--run`, it writes prompts without making API
calls. For example:

```sh
python3 code/collect.py generate --model opus46 --query c00_inadequacy --condition depression --phrasing explicit_brief
```

For fresh calls, install `boto3`, supply your own AWS credentials, choose a
region with model access and add `--run`. `judge` and `rephrase` accept `--input`
with JSONL records containing `query_text` and `response`. Generate/judge use
temperature 0.0; rephrasing uses 0.7. Anthropic generation uses 1,024 output
tokens; other providers use 4,096. The primary judge requests 256 tokens; the
recorded adapter enforces at least 4,096 for non-Anthropic providers.
Generation has no system prompt. The collector writes new files under
`outputs/` and refuses to overwrite an existing run file.

Provider model IDs are recorded; complete hosted deployment revisions were not.
New API calls need not return the archived text. The release uses final merged
records rather than duplicating the seven phase/recapture files. New collection
code provides a direct entry point for the same prompt, generation, judging
and rephrasing operations; it does not replay the historical scheduling of
phase expansion and recapture.
