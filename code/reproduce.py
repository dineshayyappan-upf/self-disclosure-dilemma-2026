#!/usr/bin/env python3
"""Reproduce paper results from the released responses and saved distances."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import behavioral
import cleaning

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['haiku45', 'sonnet46', 'opus46', 'qwen3_vl_235b', 'gpt_oss_safeguard_120b']
FOCAL = MODELS[2:]


def jsonl(name):
    with gzip.open(ROOT / name, 'rt', encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def compare(actual, expected, keys, columns=None):
    a = actual.set_index(keys).sort_index()
    e = expected.set_index(keys).sort_index()
    assert a.index.equals(e.index), 'Different result identities: ' + str(keys)
    cols = columns or [x for x in e.select_dtypes(include='number').columns if x in a]
    np.testing.assert_allclose(a[cols].to_numpy(float), e[cols].to_numpy(float),
                               atol=1e-10, rtol=1e-9, equal_nan=True)


def check_texts(records, refs):
    names = cleaning.build_group_name_regex(records)
    checked = 0
    for r in records:
        if 'analysis_text_sha256' in r:
            text = cleaning.normalized_clean_text(r, names)
            assert hashlib.sha256(text.encode()).hexdigest() == r['analysis_text_sha256']
            checked += 1
    assert checked == 29912
    for r in refs:
        for field, digest in [('original_response', 'original_analysis_sha256'),
                              ('similar_response', 'rephrasing_analysis_sha256')]:
            text = cleaning.normalized_clean_text(dict(response=r[field], query_text=r['query_text']), names)
            assert hashlib.sha256(text.encode()).hexdigest() == r[digest]
    return names


def main():
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    records = jsonl('data/responses.jsonl.gz')
    refs_text = jsonl('data/rephrasing.jsonl.gz')
    assert len(records) == 30090 and len(refs_text) == 170
    check_texts(records, refs_text)
    tags = pd.DataFrame([dict(**{k:v for k,v in r.items() if k not in {'tags', 'judge_raw'}},
                              **{t:int(r['tags'].get(t) == 'Yes') for t in behavioral.TAG_ORDER},
                              response_present=int(bool(r['response'].strip())),
                              visible_word_count=cleaning.visible_word_count(r['response'])) for r in records])
    assert tags.query_id.nunique() == 34 and set(tags.model) == set(MODELS)
    assert (tags.response_present == 0).sum() == 8
    assert not tags.duplicated(['model', 'query_id', 'condition', 'phrasing']).any()
    checks = [
        ('table1_model_tag_rates.csv', behavioral.prep_table1_model_tag_rates, ['model', 'tag']),
        ('table2_group_tag_rates.csv', behavioral.prep_table2_group_tag_rates, ['category', 'tag']),
        ('table3_depression_focus_opus.csv', behavioral.prep_table3_depression_phrasing_opus, ['focus_group', 'phrasing', 'tag']),
    ]
    for name, function, keys in checks:
        actual = function(tags)
        compare(actual, pd.read_csv(ROOT / 'data/tables' / name), keys)
        actual.to_csv(out / name, index=False)
    radar = behavioral.prep_fig4_condition_radars(tags)
    radar = radar[radar.category.isin(['type1', 'type2'])]
    compare(radar, pd.read_csv(ROOT / 'data/tables/condition_radars.csv'), ['category', 'condition', 'model', 'tag'])
    radar.to_csv(out / 'condition_radars.csv', index=False)
    lengths = pd.DataFrame([dict(model=m, questions=g.query_id.nunique(), records=len(g),
                                empty_responses=int((g.response_present == 0).sum()),
                                total_visible_words=int(g.visible_word_count.sum()),
                                mean_response_words=g.visible_word_count.mean())
                           for m,g in tags[tags.condition != 'bare'].groupby('model')])
    expected = pd.read_csv(ROOT / 'data/tables/model_response_lengths.csv')
    compare(lengths[lengths.model.isin(expected.model)], expected, ['model'])
    lengths.to_csv(out / 'model_response_lengths.csv', index=False)

    raw = pd.read_csv(ROOT / 'data/gte/response_scores.csv.gz')
    refs = pd.read_csv(ROOT / 'data/gte/rephrasing_pairs.csv')
    expected = pd.read_csv(ROOT / 'data/gte/summary_centered_distance.csv')
    assert len(raw) == 29912 and len(refs) == 170
    assert refs.groupby('model_id').size().to_dict() == {m:34 for m in MODELS}
    assert not raw.duplicated(['model_id','question_id','condition_id','phrasing_id']).any()
    expected_keys = {(r['model'], r['query_id'], r['condition'], r['phrasing']) for r in records if 'analysis_text_sha256' in r}
    assert set(zip(raw.model_id, raw.question_id, raw.condition_id, raw.phrasing_id)) == expected_keys
    c = float(refs.embedding_distance.mean())
    assert abs(c - 0.05535751899048206) < 1e-14
    np.testing.assert_allclose(raw.centered_distance, raw.embedding_distance - c, atol=1e-12, rtol=0)
    questions = sorted(raw.question_id.unique())
    result = []
    for row in expected.itertuples():
        part = raw[raw.model_id.isin(FOCAL)] if row.scope == 'c' else raw[raw.model_id == row.model_id]
        if row.category_id != 'all':
            part = part[part.category_id == row.category_id]
        if row.phrasing_id != 'all':
            part = part[part.phrasing_id == row.phrasing_id]
        sums = part.groupby('question_id').centered_distance.sum().reindex(questions).to_numpy()
        counts = part.groupby('question_id').size().reindex(questions).to_numpy()
        assert (counts > 0).all()
        draws = np.random.default_rng(int(row.bootstrap_seed)).integers(0, 34, size=(5000, 34))
        if row.scope == 'c':
            mean = sums.sum() / counts.sum()
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        else:
            qmean = sums / counts
            mean = qmean.mean()
            boot = qmean[draws].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        result.append(dict(scope=row.scope, model_id=row.model_id, category_id=row.category_id,
                           phrasing_id=row.phrasing_id, responses=len(part), questions=34,
                           mean=mean, low=lo, high=hi))
    summary = pd.DataFrame(result)
    compare(summary, expected, ['scope','model_id','category_id','phrasing_id'], ['responses','questions','mean','low','high'])
    summary.to_csv(out / 'summary_centered_distance.csv', index=False)

    items = jsonl('data/reliability_sample.jsonl.gz')
    judgments = jsonl('data/reliability_judgments.jsonl.gz')
    assert len(items) == 643 and len(judgments) == 1286
    lookup = {(r['sample_id'], r['judge_model']):r for r in judgments if r['status'] == 'ok'}
    metrics = []
    for tag in behavioral.TAG_ORDER:
        left = [lookup[(r['sample_id'], 'opus46')]['tags'][tag] for r in items]
        right = [lookup[(r['sample_id'], 'qwen3_vl_235b')]['tags'][tag] for r in items]
        n = len(items)
        agreement = float(np.mean(np.asarray(left) == np.asarray(right)))
        pe = sum(left.count(x) * right.count(x) / n**2 for x in set(left + right))
        kappa = (agreement-pe)/(1-pe) if abs(1-pe)>1e-12 else float(agreement == 1)
        metrics.append(dict(tag=tag, n=n, agreement=agreement, cohen_kappa=kappa))
    actual = pd.DataFrame(metrics)
    compare(actual, pd.read_csv(ROOT / 'data/tables/reliability.csv'), ['tag'])
    actual.to_csv(out / 'reliability.csv', index=False)
    matrix = pd.read_csv(ROOT / 'qualitative/verification_matrix.csv')
    counts = {code:int((matrix[code] == 'present').sum()) for code in ['T01a','T01b','T02','T03']}
    assert counts == dict(T01a=5, T01b=4, T02=10, T03=7)
    assert matrix.loc[matrix.case_id == 'V06', 'T03'].item() == 'unclear'
    report = dict(status='passed', primary_records=30090, empty_responses=8,
                  cleaned_tailored_texts_verified=29912, reference_texts_verified=340,
                  distance_summaries=33, behavioral_tables=3, radar_rates=450,
                  reliability_responses=643, qualitative_source_counts=counts,
                  discovery_pairs=12, verification_pairs=12,
                  fresh_model_inference=False)
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
