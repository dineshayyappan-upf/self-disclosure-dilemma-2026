#!/usr/bin/env python3
"""Optionally re-encode released texts with the pinned whole-answer encoder."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from reproduce import ROOT, jsonl, check_texts
import cleaning

MODEL = 'Alibaba-NLP/gte-modernbert-base'
REVISION = 'e7f32e3c00f91d699e8c43b53106206bcc72bb22'


def encode(model, texts):
    lengths = np.array([len(model.tokenizer(t, truncation=False)['input_ids']) for t in texts])
    assert lengths.max() <= 8192, 'An input exceeds the whole-answer context window'
    result = np.zeros((len(texts), 768), dtype=np.float32)
    order = np.argsort(lengths, kind='stable')
    done = 0
    while done < len(order):
        length = lengths[order[done]]
        size = 32 if length <= 512 else 12 if length <= 1024 else 4 if length <= 2048 else 2
        ids = order[done:done+size]
        values = model.encode([texts[i] for i in ids], batch_size=len(ids),
                              normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        assert np.isfinite(values).all()
        result[ids] = values
        done += len(ids)
        if done % 128 < len(ids) or done == len(order):
            print(f'Encoded {done}/{len(order)}', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, help='Local copy of the pinned model; otherwise download it')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32','float16'], default='float32')
    p.add_argument('--limit', type=int, help='Encode only the first N response cells as a smoke check')
    args = p.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    records = jsonl('data/responses.jsonl.gz')
    refs = jsonl('data/rephrasing.jsonl.gz')
    names = cleaning.build_group_name_regex(records) if args.limit else check_texts(records, refs)
    model = SentenceTransformer(str(args.model_path) if args.model_path else MODEL,
        revision=None if args.model_path else REVISION, local_files_only=bool(args.model_path),
        device=args.device, config_kwargs={'reference_compile':False},
        model_kwargs={'attn_implementation':'sdpa', 'dtype':getattr(torch, args.dtype)})
    model.max_seq_length = 8192
    scores = pd.read_csv(ROOT / 'data/gte/response_scores.csv.gz')
    if args.limit:
        scores = scores.head(args.limit)
    bykey = {(r['model'], r['query_id'], r['condition'], r['phrasing']):r for r in records}
    selected = [bykey[(r.model_id,r.question_id,r.condition_id,r.phrasing_id)] for r in scores.itertuples()]
    if args.limit:
        selected_pairs = {(r['model'],r['query_id']) for r in selected}
        refs = [r for r in refs if (r['model'],r['query_id']) in selected_pairs]
    original = [cleaning.normalized_clean_text(dict(response=r['original_response'],query_text=r['query_text']), names) for r in refs]
    rewrites = [cleaning.normalized_clean_text(dict(response=r['similar_response'],query_text=r['query_text']), names) for r in refs]
    tailored = [cleaning.normalized_clean_text(r, names) for r in selected]
    av = encode(model, original)
    rv = encode(model, rewrites)
    tv = encode(model, tailored)
    # The archived rephrasing calculation re-normalized vectors in float64.
    a64, r64 = av.astype(np.float64), rv.astype(np.float64)
    a64 /= np.linalg.norm(a64, axis=1, keepdims=True)
    r64 /= np.linalg.norm(r64, axis=1, keepdims=True)
    ref_dist = np.maximum(0, 1-np.einsum('ij,ij->i', a64, r64))
    lookup = {(r['model'],r['query_id']):i for i,r in enumerate(refs)}
    ai = [lookup[(r.model_id,r.question_id)] for r in scores.itertuples()]
    distances = 1-np.einsum('ij,ij->i', tv.astype(np.float64), av[ai].astype(np.float64))
    out = ROOT / 'outputs/reencoded'
    out.mkdir(parents=True, exist_ok=True)
    result = scores[['model_id','question_id','condition_id','phrasing_id']].copy()
    result['embedding_distance'] = distances
    calibration = float(pd.read_csv(ROOT / 'data/gte/rephrasing_pairs.csv').embedding_distance.mean()) if args.limit else float(ref_dist.mean())
    result['centered_distance'] = distances-calibration
    result['difference_from_archived_distance'] = distances-scores.embedding_distance.to_numpy()
    result.to_csv(out / 'response_scores.csv', index=False)
    pd.DataFrame([dict(model_id=r['model'],question_id=r['query_id'],embedding_distance=float(d))
                  for r,d in zip(refs,ref_dist)]).to_csv(out / 'rephrasing_pairs.csv', index=False)
    print('Saved fresh encoder results separately in outputs/reencoded/.')
    print('Maximum response-distance difference:', float(np.abs(result.difference_from_archived_distance).max()))


if __name__ == '__main__':
    main()
