#!/usr/bin/env python3
"""Generate, judge or rephrase selected study prompts through Amazon Bedrock.

Without --run this only writes the exact planned prompts. New records are saved
under outputs/ and never replace the archive.
"""
import argparse
import json
import re
from pathlib import Path
from bedrock import bedrock_invoke, strip_reasoning_tags
ROOT = Path(__file__).resolve().parents[1]


def parse_tags(text, tags):
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            value = json.loads(match.group())
            return {t:'Yes' if value.get(t,'No') in ('Yes','yes',True) else 'No' for t in tags}
        except json.JSONDecodeError:
            pass
    result = {}
    for tag in tags:
        match = re.search(rf'{tag.upper()}[:\s]*(Yes|No)', text, re.I)
        result[tag] = match.group(1).capitalize() if match else 'No'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['generate','judge','rephrase'])
    p.add_argument('--model', required=True)
    p.add_argument('--query', help='Study query ID (required for generate)')
    p.add_argument('--condition', default='bare')
    p.add_argument('--phrasing', default='explicit_brief')
    p.add_argument('--input', type=Path, help='JSONL records with query_text and response, for judge/rephrase')
    p.add_argument('--region', default='eu-west-1')
    p.add_argument('--run', action='store_true', help='Make paid API calls using your own AWS credentials')
    args = p.parse_args()
    instruments = json.loads((ROOT / 'instruments/prompts_and_rubric.json').read_text())
    settings = json.loads((ROOT / 'instruments/collection_settings.json').read_text())
    if args.model not in settings['models']:
        p.error('Choose a model key listed in instruments/collection_settings.json')
    cfg = settings['models'][args.model]
    if args.action == 'generate':
        if not args.query:
            p.error('--query is required')
        q = next(r for r in instruments['queries'] if r['query_id'] == args.query)
        disclosure = '' if args.condition == 'bare' else next(r['disclosure_text'] for r in instruments['collected_disclosures'] if r['condition'] == args.condition and r['phrasing'] == args.phrasing)
        records = [dict(query_text=q['query_text'], query_id=q['query_id'], condition=args.condition,
                        phrasing='none' if args.condition == 'bare' else args.phrasing,
                        disclosure_text=disclosure, prompt=(disclosure+' '+q['query_text']).strip())]
        temperature, max_tokens = 0.0, 1024 if cfg['provider'] == 'anthropic' else 4096
    else:
        if not args.input:
            p.error('--input is required')
        records = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
        for r in records:
            if args.action == 'judge':
                r['prompt'] = instruments['judge_prompt'].format(user_query=r['query_text'], response=r['response'])
            else:
                r['prompt'] = (ROOT / 'instruments/rephrasing_prompt.txt').read_text().format(query_text=r['query_text'], response=r['response'])
        temperature = 0.0 if args.action == 'judge' else 0.7
        max_tokens = 256 if args.action == 'judge' else 4096
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    target = out / (args.action + ('_new' if args.run else '_prompts') + '.jsonl')
    # Exclusive creation prevents accidentally overwriting a previous run.
    with target.open('x', encoding='utf-8') as f:
        if args.run:
            import boto3
            client = boto3.client('bedrock-runtime', region_name=args.region)
        for r in records:
            result = dict(r)
            result['model' if args.action == 'generate' else 'judge_model' if args.action == 'judge' else 'rephrasing_model'] = args.model
            if args.run:
                text, _ = bedrock_invoke(client, cfg['id'], [{'role':'user','content':r['prompt']}],
                                         provider=cfg['provider'], temperature=temperature, max_tokens=max_tokens)
                result['response' if args.action == 'generate' else 'judge_raw' if args.action == 'judge' else 'similar_response'] = text if cfg['provider'] == 'anthropic' else strip_reasoning_tags(text)
                if args.action == 'judge':
                    result['tags'] = parse_tags(text, [r['tag'] for r in instruments['behavioral_tags']])
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
    print('Saved', len(records), 'records to', target.relative_to(ROOT))


if __name__ == '__main__':
    main()
