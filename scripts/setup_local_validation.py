"""Prepare explicit SILT user-workflow fixtures. Does not run models or manufacture results."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--models-root',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.models_root).resolve(); out=Path(a.output).resolve()
if out.exists():
    raise SystemExit('Output already exists; refusing to overwrite fixtures')
out.mkdir(parents=True)

def component(kind,task,name,license='Apache-2.0'):
    return {'kind':kind,'task':task,'model_path':str(root/name),'risk':'low','provenance':[{'source':name+'; exact acquisition.json accompanies local weights','license':license,'risk':'low'}]}
def node(id,source,c,prefix='',tokens=128):
    return {'id':id,'input':source,'component':c,'prompt_prefix':prefix,'max_new_tokens':tokens}
def spec(name,input_type,nodes):
    return {'schema_version':1,'name':name,'input_type':input_type,'nodes':nodes,'output_node':nodes[-1]['id'],'limits':{'max_seconds':300.0,'max_peak_rss_mb':3500.0,'threads':2}}
def put(name,value):
    (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
q=component('hf_text','causal','Qwen2.5-Coder-0.5B-Instruct')
s=component('hf_text','causal','SmolLM2-135M-Instruct')
asr=component('hf_asr','whisper','whisper-tiny.en')
tts=component('hf_tts','vits','mms-tts-eng','CC-BY-NC-4.0; noncommercial experiment only; not redistributed')
v=component('hf_vision','smolvlm','SmolVLM-256M-Instruct')
prefix='Write only the requested Python function. No examples, explanations, imports, or print statements.\n'
put('coder.json',spec('qwen-coding-candidate','text',[node('coder','$input',q,prefix)]))
put('small-coder.json',spec('smollm-coding-baseline','text',[node('coder','$input',s,prefix)]))
put('tts.json',spec('tts-mechanism-noncommercial','text',[node('voice','$input',tts)]))
put('asr.json',spec('whisper-asr-candidate','audio',[node('listener','$input',asr)]))
put('voice-assistant.json',spec('voice-composition-candidate','audio',[node('listener','$input',asr),node('assistant','listener',s,'Answer the request briefly in plain English.\n',64),node('voice','assistant',tts)]))
put('vision.json',spec('matched-smolvlm-candidate','image',[node('vision','$input',v,tokens=64)]))
put('vision-coder.json',spec('vision-observation-to-coder','image',[node('vision','$input',v,'Read the visible error and describe it. ',64),node('coder','vision',q,'Explain this observed error and suggest a fix:\n',128)]))
# Tiny diagnostic tasks only, not a public benchmark or evidence of broad coding skill.
examples=[
('add','target','Define add(a, b), returning their sum.','self.assertEqual(candidate.add(2,3),5)\n        self.assertEqual(candidate.add(-2,2),0)'),
('square','target','Define square(x), returning x multiplied by itself.','self.assertEqual(candidate.square(4),16)\n        self.assertEqual(candidate.square(-3),9)'),
('even','target','Define is_even(n), returning True for an even integer and False otherwise.','self.assertIs(candidate.is_even(0),True)\n        self.assertIs(candidate.is_even(3),False)'),
('reverse','target','Define reverse_text(s), returning the string reversed.','self.assertEqual(candidate.reverse_text("abc"),"cba")\n        self.assertEqual(candidate.reverse_text(""),"")'),
('maximum','control','Define larger(a, b), returning the larger value, including when both are negative.','self.assertEqual(candidate.larger(-4,-2),-2)\n        self.assertEqual(candidate.larger(5,5),5)'),
('absolute','control','Define absolute_value(n), returning its absolute value.','self.assertEqual(candidate.absolute_value(-7),7)\n        self.assertEqual(candidate.absolute_value(0),0)')]
cases=[]
for id,group,prompt,tests in examples:
    reference='import unittest\nimport candidate\nclass Checks(unittest.TestCase):\n    def test_values(self):\n        '+tests+'\n'
    cases.append({'id':id,'group':group,'input':prompt,'reference':reference,'metric':'functional_code','threshold':1.0})
put('coding-diagnostic.json',{'schema_version':1,'name':'six-function-diagnostic-not-general-benchmark','reference_source':'Predeclared hand-authored unit checks; no model answers used as labels. Not a heldout population benchmark.','claims':['coding'],'cases':cases})
cal=[('A dog has <extra_id_0> legs.','<extra_id_0> four <extra_id_1>'),('The opposite of hot is <extra_id_0>.','<extra_id_0> cold <extra_id_1>'),('Water freezes at zero degrees <extra_id_0>.','<extra_id_0> Celsius <extra_id_1>'),('A week has seven <extra_id_0>.','<extra_id_0> days <extra_id_1>'),('The sky is <extra_id_0> on a clear day.','<extra_id_0> blue <extra_id_1>'),('Birds can <extra_id_0>.','<extra_id_0> fly <extra_id_1>')]
put('switch-calibration.json',{'samples':[{'prompt':x,'target':y} for x,y in cal]})
hold=[('France','The capital of France is <extra_id_0>.',['Paris','<extra_id_0> Paris <extra_id_1>']),('two','Two plus two equals <extra_id_0>.',['four','4','<extra_id_0> four <extra_id_1>']),('sun','The sun rises in the <extra_id_0>.',['east','<extra_id_0> east <extra_id_1>']),('winter','The season after autumn is <extra_id_0>.',['winter','<extra_id_0> winter <extra_id_1>'])]
put('switch-diagnostic.json',{'split':'heldout','cases':[{'id':i,'prompt':x,'references':y} for i,x,y in hold]})
print(json.dumps({'fixtures':str(out),'scope':'small diagnostic fixtures only; not performance evidence'}))
