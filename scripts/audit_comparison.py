#!/usr/bin/env python3
"""Independent scalar scoring and pairwise integrity audit of the two real pilots.

Does not call nir_eval.scoring, any model, or the network. Canonical gold remains unchanged.
"""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def independently_parse(text):
    if not isinstance(text,str): return None
    text=text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text=text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):
        text=text[3:-3].strip()
    try:
        obj=json.loads(text)
        if not isinstance(obj,dict): return None
        value=obj['answer']
        if isinstance(value,bool): return None
        if isinstance(value,str):
            value=value.strip().lower()
            if value in ('yes','no'):return value
            percent=value.endswith('%')
            if percent:value=value[:-1].strip()
            value=value.replace(',','')
            if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',value):return None
            value=float(value)/(100 if percent else 1)
        elif isinstance(value,(float,int)):value=float(value)
        else:return None
        return value if math.isfinite(value) else None
    except (ValueError,KeyError,TypeError):return None


def main():
    data=json.loads((ROOT/'data/finqa_dev.json').read_text())
    gold={r['id']:r['qa']['exe_ans'] for r in data}
    groups={}
    by_size={}
    for size,folder in [('0.5B','qwen_0_5b'),('7B','qwen_7b')]:
        source=ROOT/'evidence'/folder
        manifest=json.loads((source/'manifest.json').read_text())
        raw=(source/'records.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest['records_sha256']
        rows=[json.loads(line) for line in raw.splitlines()]
        assert len(rows)==24
        keys={(r['id'],r['config'],r['repetition']) for r in rows}
        assert len(keys)==24
        official=json.loads((ROOT/'runs'/('qwen_0_5b_finqa_pilot' if size=='0.5B' else 'qwen_7b_finqa_pilot')/'summary.json').read_text())
        by_size[size]={(r['id'],r['config']):r for r in rows}
        for protocol in ['direct','cot']:
            subset=[r for r in rows if r['config']==protocol]
            answers=[independently_parse(r['raw_output']) if not r.get('error') else None for r in subset]
            flags=[a is not None and ((a==gold[r['id']]) if isinstance(a,str) else round(a,5)==gold[r['id']])
                   for a,r in zip(answers,subset)]
            matched=next(g for g in official['groups'] if g['config']==protocol)
            assert sum(flags)==matched['correct']
            assert sum(a is None for a in answers)==matched['parse_errors']+matched['provider_errors']
            groups[size+'/'+protocol]={'n':len(subset),'correct':sum(flags),'parseable':sum(a is not None for a in answers),
                                      'correct_ids':[r['id'] for r,flag in zip(subset,flags) if flag]}
    assert by_size['0.5B'].keys()==by_size['7B'].keys()
    equal_messages=all(by_size['0.5B'][k]['messages']==by_size['7B'][k]['messages'] for k in by_size['0.5B'])
    equal_renderings=all(by_size['0.5B'][k]['rendered_prompt']==by_size['7B'][k]['rendered_prompt'] for k in by_size['0.5B'])
    assert equal_messages and equal_renderings
    comparisons=[]
    for protocol in ['direct','cot']:
        a=set(groups['0.5B/'+protocol]['correct_ids']);b=set(groups['7B/'+protocol]['correct_ids'])
        a_only=len(a-b);b_only=len(b-a);discordant=a_only+b_only
        p=min(1.0,2*sum(math.comb(discordant,i) for i in range(min(a_only,b_only)+1))/2**discordant) if discordant else 1.0
        comparisons.append({'protocol':protocol,'0.5B_only_correct':a_only,'7B_only_correct':b_only,
                            'mcnemar_exact_two_sided_p':p,'claim':'Small descriptive pilot; no established general superiority.'})
    audit={'independent_scorer_matches_mvp':True,'records':48,'same_12_ids':True,
           'same_messages_all_24_pairs':equal_messages,'same_rendered_prompts_all_24_pairs':equal_renderings,
           'groups':groups,'paired_cross_model_checks':comparisons,
           'latency_comparison_warning':'0.5B measures model.generate only; 7B measures whole llama.cpp completion. Size, quantization and backend differ. Do not infer a controlled speed ratio.',
           'model_inference_performed_by_audit':False}
    destination=ROOT/'runs/model_comparison/independent_audit.json'
    destination.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
