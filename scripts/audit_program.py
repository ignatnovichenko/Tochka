#!/usr/bin/env python3
"""Cross-check the real DSL pilot with the unmodified official FinQA evaluator.
Optional dependencies: NumPy and SymPy. No model or network calls.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nir_eval.scoring import parse_object,parse_program,program_tokens


def main():
    import numpy,sympy
    shim=importlib.util.find_spec('tqdm') is None
    if shim:
        module=types.ModuleType('tqdm');module.tqdm=lambda x,*args,**kwargs:x
        sys.modules['tqdm']=module
    path=ROOT/'evidence/finqa_official_evaluate.py'
    spec=importlib.util.spec_from_file_location('finqa_official',path)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    data={r['id']:r for r in json.loads((ROOT/'data/finqa_dev.json').read_text())}
    rows=[json.loads(line) for line in (ROOT/'runs/qwen_7b_finqa_program_pilot/records.jsonl').read_text().splitlines()]
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    checks=[]
    for row in rows:
        gold=data[row['id']]
        result={'id':row['id'],'valid_dsl':False,'official_execution_correct':False,'official_program_correct':False}
        try:
            obj=parse_object(row['raw_output']);program=obj['program']
            parse_program(program)
            predicted=program_tokens(program)
        except (ValueError,KeyError,TypeError):
            checks.append(result);continue
        invalid,value=official.eval_program(predicted,gold['table'])
        eq_program=official.equal_program(official.program_tokenization(gold['qa']['program']),predicted)
        result.update(valid_dsl=True,official_result=value,
                      official_execution_correct=not invalid and value==gold['qa']['exe_ans'],
                      official_program_correct=bool(eq_program))
        assert value==row['parsed_answer'] and not invalid
        assert result['official_execution_correct']==row['correct']
        checks.append(result)
    report={'scope':'Two valid DSL outputs executed by unchanged official evaluator; invalid outputs remain failures in denominator12.',
            'n':12,'valid_dsl':sum(r['valid_dsl'] for r in checks),
            'official_execution_correct':sum(r['official_execution_correct'] for r in checks),
            'official_program_correct':sum(r['official_program_correct'] for r in checks),
            'valid_results_match_bounded_interpreter':True,
            'official_evaluator_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'numpy_version':numpy.__version__,'sympy_version':sympy.__version__,
            'unused_tqdm_import_shim':shim,'model_inference_performed_by_audit':False,'cases':checks}
    assert report['valid_dsl']==2 and report['official_execution_correct']==1 and report['official_program_correct']==1
    destination=ROOT/'runs/model_comparison_with_program/official_program_audit.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
