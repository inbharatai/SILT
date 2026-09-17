"""CPU-only tests. All fabricated hardware/model profiles are explicitly synthetic.

Large dimension tests contain JSON configuration only, never large tensors/files.
"""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from asea.artifacts import digest
from asea.hardware import probe_hardware, plan_specialist
from asea.hardware.metadata import _qwen_shapes, _switch_shapes, shape_metadata
from asea.hardware.probe import _linux_resources

GiB = 1024**3


def qwen_config(h=32, width=128, layers=2, vocab=128, heads=4, kv=2, tied=True):
    return {'model_type':'qwen2','hidden_size':h,'intermediate_size':width,
            'num_hidden_layers':layers,'vocab_size':vocab,'num_attention_heads':heads,
            'num_key_value_heads':kv,'tie_word_embeddings':tied,'hidden_act':'silu'}


def write_checkpoint(root, config, dtype='BF16', bound=True, overrides=None):
    root.mkdir()
    (root/'config.json').write_text(json.dumps(config))
    (root/'tokenizer.json').write_text('{}')  # config-only fixtures never instantiate a tokenizer
    if not bound:
        return root
    write_tiny_tokenizer(root)
    shapes = (_qwen_shapes(config,.75) if config['model_type']=='qwen2' else _switch_shapes(config))[0]
    header, cursor = {}, 0
    sizes = {'BF16':2,'F32':4,'F16':2}
    for name, shape in sorted(shapes.items()):
        dt = (overrides or {}).get(name,dtype)
        size = math.prod(shape)*sizes[dt]
        header[name] = {'dtype':dt,'shape':shape,'data_offsets':[cursor,cursor+size]}
        cursor += size
    raw = json.dumps(header,separators=(',',':')).encode()
    raw += b' '*((-len(raw))%8)
    # Only tiny fixtures allocate payload (largest is <1 MiB).
    assert cursor < 1024**2
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*cursor)
    return root


def write_tiny_tokenizer(root):
    """Real local tokenizer, synthetic plain-language data; no model run or quality claim."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({'<unk>':0,'<eos>':1,'alpha':2,'beta':3,'gamma':4,'delta':5,
                                     'one':6,'two':7,'three':8,'four':9}, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(single='$A <eos>', special_tokens=[('<eos>',1)])
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='<unk>', eos_token='<eos>', pad_token='<unk>').save_pretrained(root)


def write_permitted_data(tmp_path, train=None, dev=None):
    from asea.artifacts import file_hash
    train = train if train is not None else [dict(id='t1',family='train-a',prompt='alpha',response='one'),
                                           dict(id='t2',family='train-b',prompt='beta',response='two')]
    dev = dev if dev is not None else [dict(id='v1',family='dev-a',prompt='gamma',response='three'),
                                     dict(id='v2',family='dev-b',prompt='delta',response='four')]
    names = {'training':'train.json','validation_data':'validation.json','calibration':'calibration.json',
             'validation_suite':'validation-suite.json','data_manifest':'manifest.json','selection_lock':'selection-lock.json'}
    for name, rows in (('train.json',train),('validation.json',dev)):
        (tmp_path/name).write_text(json.dumps({'samples':rows}))
    for name in ('calibration.json','final-suite.json'):
        (tmp_path/name).write_text('SENTINEL: DO NOT OPEN OR HASH SUITE CASES / FINAL ANSWERS')
    suite = {'name':'synthetic governed DEV', 'reference_source':'synthetic fixtures only', 'claims':['coding'],
        'cases':[{'id':r['id'], 'group':'target' if i==0 else 'control', 'input':r['prompt'],
                  'reference':'not tokenized', 'metric':'function_io', 'threshold':1.0,
                  'function_cases':[{'id':'one','function':'f','args':[1],'kwargs':{},'expected':1}]}
                 for i,r in enumerate(dev)]}
    (tmp_path/'validation-suite.json').write_text(json.dumps(suite))
    selection = [dict(id=r['id'],family=r['family'],split=s) for s,rows in (('train',train),('validation',dev)) for r in rows]
    selection.append(dict(id='heldout-only-id',family='heldout-only-family',split='final'))
    lock = dict(schema='silt.specialist.selection-lock.v1',frozen_before_model_generation=True,
                model_outputs_consulted=False,selection=selection,prior_consumed_selection=[])
    (tmp_path/'selection-lock.json').write_text(json.dumps(lock))
    manifest = dict(schema='silt.specialist.manifest.v1',selection_lock_sha256=file_hash(tmp_path/'selection-lock.json')['sha256'],
                    artifact_sha256={name:file_hash(tmp_path/name)['sha256'] for name in ('train.json','validation.json','validation-suite.json')})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return {key:str(tmp_path/name) for key,name in names.items()}


def recipe(tmp_path, config=None, bound=True):
    source = write_checkpoint(tmp_path/'model',config or qwen_config(),bound=bound)
    data = write_permitted_data(tmp_path)
    return {'source_path':str(source),**data,
            'dtype':'bfloat16','max_length':16,'max_new_tokens':8,'output_budget_bytes':32*GiB,
            'reconstruction':{'retention':.75},'recovery':{'rank':4,'max_samples':2,'export_mode':'factor_preserving'}}


def profile(parent, ram=64*GiB, devices=None, disk=128*GiB):
    return {'schema_version':1,'profile_kind':'synthetic','platform':{'system':'Linux','wsl':False},
            'cpu':{'effective_cores':2},'memory':{'available_bytes':ram,'host_total_bytes':ram},
            'disks':[{'path':str(parent),'free_bytes':disk,'total_bytes':disk}],
            'torch':{'status':'ok','torch_version':'synthetic','cuda_version':'synthetic' if devices else None,
                'hip_version':None,'cpu':{'float32':True,'bfloat16':True},'mps_available':False,
                'packages':{name:'synthetic' for name in ('torch','transformers','peft','accelerate','safetensors')},
                'devices':devices or []}}


def gpu(index=0,free=48*GiB,bf16=True):
    return {'index':index,'name':'SYNTHETIC capability profile, not detected hardware','vendor':'NVIDIA',
            'total_bytes':free,'free_bytes':free,'allocated_bytes':0,'reserved_bytes':0,'bf16_supported':bf16,'probe_valid':True,
            'primitives':{'float32':True,'bfloat16':True}}


def test_import_is_lightweight():
    code = 'import sys; import asea.hardware; assert not any(x in sys.modules for x in ("torch","transformers","peft"))'
    env = __import__('os').environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]/'src')
    subprocess.run([sys.executable,'-c',code],env=env,check=True)


def test_cpu_ready_uses_actual_headers(tmp_path):
    cfg = recipe(tmp_path)
    p = plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='READY',p['reasons']
    assert p['execution_device']=='cpu'
    assert p['model']['checkpoint_bound'] and p['model']['counts']['bound_to_headers']
    assert p['no_execution'] and not p['quality_approval'] and not p['reservation']
    assert 'synthetic' in p['evidence_level']
    assert p['bindings']['data_metadata']['training']['contents_read']
    assert p['bindings']['data_metadata']['validation_data']['contents_read']
    assert not p['bindings']['data_metadata']['calibration']['contents_read']
    assert p['bindings']['data_metadata']['validation_suite']['contents_read']
    assert p['data_profile']['_metadata']['final_opened'] is False
    assert p['plan_sha256']==digest({k:v for k,v in p.items() if k!='plan_sha256'})


def test_small_ram_blocks_large_ram_not_fixed_4gb(tmp_path):
    cfg = recipe(tmp_path)
    small = plan_specialist(cfg,profile(tmp_path,ram=256*1024**2),workspace_parent=tmp_path)
    large = plan_specialist(cfg,profile(tmp_path,ram=16*GiB),workspace_parent=tmp_path)
    assert small['status']=='BLOCKED' and any('host_ram' in r for r in small['reasons'])
    assert large['status']=='READY'


def test_exact_memory_operator_cap(tmp_path, monkeypatch):
    from asea.specialist import evaluation
    # Stable simulated interpreter baseline, not a hardware measurement claim.
    monkeypatch.setattr(evaluation, '_current_rss', lambda: 128*1024**2)
    cfg = recipe(tmp_path)
    initial = plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    peak = max(initial['estimates']['cpu_peak_bytes'],
        initial['estimates']['evaluation_host_operator_budget']['estimated_total_process_peak_bytes'])
    cfg['memory_budget_bytes'] = peak
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='READY'
    cfg['memory_budget_bytes'] = peak-1
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='BLOCKED'


def test_exact_output_budget_and_no_mutation(tmp_path):
    cfg = recipe(tmp_path)
    initial = plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    required = initial['estimates']['output_required_bytes']
    cfg['output_budget_bytes']=required
    saved=copy.deepcopy(cfg)
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='READY'
    assert cfg==saved
    cfg['output_budget_bytes']=required-1
    blocked=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert blocked['status']=='BLOCKED' and blocked['estimates']['proposed_output_budget_bytes']==required
    assert cfg['output_budget_bytes']==required-1


@pytest.mark.parametrize('h,width,layers,heads,kv,tied',[
    (1536,8960,28,12,2,True), (2048,11008,36,16,2,True), (3584,18944,28,28,4,False)])
def test_2_3_7b_dimensions_config_only_never_ready(tmp_path,h,width,layers,heads,kv,tied):
    cfg=recipe(tmp_path,qwen_config(h,width,layers,151936,heads,kv,tied),bound=False)
    p=plan_specialist(cfg,profile(tmp_path,ram=1024*GiB,disk=1024*GiB),workspace_parent=tmp_path)
    assert p['status']!='READY' and p['execution_device'] is None
    assert not p['model']['counts']['bound_to_headers']
    assert any('no_source_checkpoint' in r for r in p['reasons'])
    assert (tmp_path/'model'/'model.safetensors').exists() is False


def test_qwen3b_4gib_base_exact_before_any_model_load(tmp_path):
    cfg=recipe(tmp_path,qwen_config(2048,11008,36,151936,16,2,True),bound=False)
    cfg['output_budget_bytes']=4*GiB
    p=plan_specialist(cfg,profile(tmp_path,ram=1024*GiB,disk=1024*GiB),workspace_parent=tmp_path)
    assert p['model']['counts']['base_tensor_bytes']==4954480640
    assert p['model']['counts']['base_parameters']==2477240320
    assert p['model']['counts']['reduction']['output_intermediate_size']==8256
    assert p['status']=='BLOCKED' and any('output_budget_exceeded' in r for r in p['reasons'])
    assert p['estimates']['proposed_output_budget_bytes']>4*GiB


def test_f32_raw_bytes_and_native_cast_windows(tmp_path):
    cfg=recipe(tmp_path)
    (tmp_path/'model'/'model.safetensors').unlink()
    # Rewrite tiny source as F32 while preserving requested BF16 (explicit recipe).
    other=write_checkpoint(tmp_path/'f32',qwen_config(),dtype='F32')
    cfg['source_path']=str(other)
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    c=p['model']['counts']
    assert p['status']=='READY'
    assert c['source_raw_tensor_bytes']==4*c['source_parameters']
    assert c['source_loaded_tensor_bytes']==2*c['source_parameters']
    assert c['source_cast_pages_bytes']==c['source_raw_tensor_bytes']
    assert c['source_cast_scratch_bytes']>0
    cfg['dtype']='float32'
    p2=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p2['model']['counts']['base_tensor_bytes']==2*c['base_tensor_bytes']


@pytest.mark.parametrize('bad',['float16','BF16','auto',None,{},True])
def test_invalid_dtype_refused(tmp_path,bad):
    cfg=recipe(tmp_path)
    cfg['dtype']=bad
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='BLOCKED'


def test_cgroup_ancestor_remaining_and_cpu_intersection():
    files={'/proc/meminfo':'MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n',
        '/proc/self/cgroup':'0::/tenant/work\n',
        '/proc/self/mountinfo':'1 0 0:1 /tenant /cg rw - cgroup2 cgroup rw\n',
        '/cg/work/memory.max':str(8*GiB),'/cg/work/memory.current':str(3*GiB),
        '/cg/memory.max':str(16*GiB),'/cg/memory.current':str(14*GiB),
        '/cg/work/cpu.max':'300000 100000','/cg/cpu.max':'150000 100000',
        '/cg/work/cpuset.cpus.effective':'0-3','/cg/cpuset.cpus.effective':'2-7'}
    def read(p):
        if str(p) not in files: raise FileNotFoundError(str(p))
        return files[str(p)]
    r=_linux_resources(read,cpu_count=16,affinity={1,2,3,4})
    assert r['memory']['available_bytes']==2*GiB
    assert r['cpu']['cpuset_intersection_count']==2 and r['cpu']['effective_cores']==1.5


def test_cgroup_v1_and_v2_constraints_both_apply():
    files={'/proc/meminfo':'MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n',
        '/proc/self/cgroup':'0::/\n3:memory:/\n2:cpu,cpuacct:/\n',
        '/proc/self/mountinfo':'1 0 0:1 / /v2 rw - cgroup2 cgroup rw\n2 0 0:2 / /v1 rw - cgroup cgroup rw,memory\n3 0 0:3 / /cpu rw - cgroup cgroup rw,cpu,cpuacct\n',
        '/v2/memory.max':str(10*GiB),'/v2/memory.current':str(2*GiB),
        '/v1/memory.limit_in_bytes':str(5*GiB),'/v1/memory.usage_in_bytes':str(4*GiB),
        '/v2/cpu.max':'400000 100000','/cpu/cpu.cfs_quota_us':'200000','/cpu/cpu.cfs_period_us':'100000'}
    def read(p):
        if str(p) not in files: raise FileNotFoundError(str(p))
        return files[str(p)]
    r=_linux_resources(read,cpu_count=16)
    assert r['memory']['available_bytes']==GiB and r['cpu']['effective_cores']==2
    files['/v1/memory.usage_in_bytes']='bad'
    assert _linux_resources(read,cpu_count=16)['memory']['available_bytes'] is None


def test_auto_cuda_capability_not_model_name(tmp_path):
    cfg=recipe(tmp_path)
    p=plan_specialist(cfg,profile(tmp_path,devices=[gpu(3)]),workspace_parent=tmp_path)
    assert p['status']=='READY' and p['execution_device']=='cuda:3'
    assert next(x for x in p['phases'] if x['name']=='reconstruction')['execution_device']=='cpu'
    assert any(x['device_ram_bytes']>0 for x in p['phases'])


def test_cuda_cannot_replace_host_ram(tmp_path):
    cfg=recipe(tmp_path)
    p=plan_specialist(cfg,profile(tmp_path,ram=256*1024**2,devices=[gpu(free=512*GiB)]),workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and p['execution_device'] is None


def test_bf16_unsupported_auto_cpu_but_explicit_blocks(tmp_path):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path,devices=[gpu(bf16=False)])
    assert plan_specialist(cfg,hw,workspace_parent=tmp_path)['execution_device']=='cpu'
    explicit=plan_specialist(cfg,hw,workspace_parent=tmp_path,requested_device='cuda:0')
    assert explicit['status']=='BLOCKED' and explicit['execution_device'] is None
    cfg['dtype']='float32'
    assert plan_specialist(cfg,hw,workspace_parent=tmp_path,requested_device='cuda:0')['execution_device']=='cuda:0'


@pytest.mark.parametrize('device',['cuda','cuda:8','mps','rocm','cuda:0,cuda:1','auto-offload','cpu:0'])
def test_explicit_unsupported_or_absent_device_never_falls_back(tmp_path,device):
    cfg=recipe(tmp_path)
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path,requested_device=device)
    assert p['status']=='BLOCKED' and p['execution_device'] is None


@pytest.mark.parametrize('field,value',[('status','timeout'),('hip_version','synthetic_rocm')])
def test_unknown_and_rocm_refuse_cuda(tmp_path,field,value):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path,devices=[gpu()])
    hw['torch'][field]=value
    p=plan_specialist(cfg,hw,workspace_parent=tmp_path,requested_device='cuda:0')
    assert p['status']=='BLOCKED'


def test_disk_must_match_actual_output_parent(tmp_path):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path)
    hw['disks'][0]['path']=str(tmp_path/'model')
    p=plan_specialist(cfg,hw,workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and 'insufficient_or_unknown_output_volume_disk' in p['reasons']


def test_missing_source_or_unknown_architecture_refused(tmp_path):
    cfg=recipe(tmp_path)
    cfg['source_path']=str(tmp_path/'absent')
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='BLOCKED'
    cfg['source_path']=str(tmp_path/'model')
    (tmp_path/'model'/'config.json').write_text(json.dumps({'model_type':'unknown-name-3B'}))
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='BLOCKED'


def test_bound_missing_tensor_or_wrong_shape_is_not_ready(tmp_path):
    cfg=recipe(tmp_path)
    path=tmp_path/'model'/'config.json'
    config=json.loads(path.read_text())
    config['vocab_size']+=1
    path.write_text(json.dumps(config))
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and any('shape mismatch' in s for s in p['reasons'])


def test_config_only_planning_only_even_when_all_estimates_fit(tmp_path):
    cfg=recipe(tmp_path,bound=False)
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='PLANNING_ONLY' and p['execution_device'] is None


def test_resident_honored_and_cached_bank_disk_explicit(tmp_path):
    cfg=recipe(tmp_path)
    a=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    cfg['recovery']['teacher_mode']='resident'
    b=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    aa=next(p for p in a['phases'] if p['name']=='adaptation')
    bb=next(p for p in b['phases'] if p['name']=='adaptation')
    assert bb['host_ram_bytes']>aa['host_ram_bytes']
    assert a['estimates']['cached_teacher_bank_disk_bytes']>0
    assert b['estimates']['cached_teacher_bank_disk_bytes']==0
    assert cfg['recovery']['teacher_mode']=='resident'


def test_strict_recipe_path_and_unknown_fields(tmp_path):
    cfg=recipe(tmp_path)
    path=tmp_path/'recipe.json'
    path.write_text(json.dumps(cfg))
    assert plan_specialist(path,profile(tmp_path),workspace_parent=tmp_path)['status']=='READY'
    cfg['quantization']='auto'
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='BLOCKED'


def test_windows_mps_discovery_not_execution(tmp_path):
    cfg=recipe(tmp_path)
    for system in ('Windows','Darwin'):
        hw=profile(tmp_path)
        hw['platform']['system']=system
        hw['torch']['mps_available']=True
        p=plan_specialist(cfg,hw,workspace_parent=tmp_path)
        assert p['status']=='BLOCKED' and p['execution_device'] is None


def switch_config():
    return {'model_type':'switch_transformers','d_model':16,'d_ff':32,'d_kv':4,'num_heads':4,
            'num_layers':2,'num_decoder_layers':2,'num_sparse_encoder_layers':1,'num_sparse_decoder_layers':1,
            'vocab_size':64,'num_experts':2,'relative_attention_num_buckets':8,'router_dtype':'float32',
            'tie_word_embeddings':True,'dense_act_fn':'relu','router_bias':True}


def test_switch_f32_exceptions_and_fixed_top1(tmp_path):
    cfg=recipe(tmp_path,switch_config())
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='READY',p['reasons']
    c=p['model']['counts']
    assert c['base_tensor_bytes']>c['base_parameters']*2
    assert c['retained_f32_parameters']>0
    assert c['reduction']['experts_per_sparse_layer']==1
    assert any('.router.classifier.' in key for key in c['dtype_exceptions'])


def test_shape_math_against_actual_tiny_qwen_reconstruction():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from asea.specialist.reconstruction import _qwen
    config=qwen_config()
    model=Qwen2ForCausalLM(Qwen2Config(**config)).to(torch.bfloat16)
    source_parameters=sum(p.numel() for p in model.parameters())
    model,*_= _qwen(model,None,[],'uniform',.75,16,torch)
    counts=shape_metadata(config,.75,'bfloat16',4)
    assert counts['source_parameters']==source_parameters
    assert counts['base_parameters']==sum(p.numel() for p in model.parameters())
    assert counts['base_tensor_bytes']==sum(p.numel()*p.element_size() for p in model.parameters())


def test_shape_math_against_actual_tiny_switch_reconstruction():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
    from asea.specialist.reconstruction import _switch
    config=switch_config()
    model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**config))
    source_parameters=sum(p.numel() for p in model.parameters())
    converted,*_=_switch(model,None,[],'uniform',16,17,torch)
    counts=shape_metadata(config,.75,'float32',4)
    assert counts['source_parameters']==source_parameters
    assert counts['base_parameters']==sum(p.numel() for p in converted.parameters())
    assert counts['base_tensor_bytes']==sum(p.numel()*p.element_size() for p in converted.parameters())


def test_selected_python_timeout_is_bounded(monkeypatch):
    from asea.hardware import probe
    class Process:
        returncode=None
        stdout=__import__('io').BytesIO(b'')
        def wait(self,timeout):
            if self.returncode is None: raise subprocess.TimeoutExpired('synthetic',timeout)
            return self.returncode
        def kill(self): self.returncode=-9
    monkeypatch.setattr(probe.subprocess,'Popen',lambda *a,**kw:Process())
    assert probe._torch_probe('/synthetic/python')['status']=='timeout'


def test_probe_without_torch_keeps_host_observations(monkeypatch,tmp_path):
    from asea.hardware import probe
    monkeypatch.setattr(probe,'_torch_probe',lambda p:{'status':'unavailable','devices':[]})
    observed=probe_hardware('/not-a-python',paths=(tmp_path/'future'/'output',))
    assert observed['profile_kind']=='detected'
    assert observed['disks'][0]['existing_parent']==str(tmp_path)
    assert observed['disks'][0]['free_bytes']>0
    assert observed['torch']['status']=='unavailable'
    assert not any(key in observed for key in ('hostname','uuid','serial','credentials'))


def test_native_switch_bf16_exception_math_matches_actual_load_plan():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
    from asea.specialist.reconstruction import _load_plan
    config=switch_config()
    model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**config))
    headers={name:{'shape':list(p.shape),'numel':p.numel(),'dtype':'F32','file':'model.safetensors'}
             for name,p in model.named_parameters()}
    native=_load_plan(model,headers,'bfloat16')
    actual=sum(entry['numel']*(4 if native['target_dtypes'][key]=='float32' else 2)
               for key,entry in headers.items())
    calculated=shape_metadata(config,.75,'bfloat16',4,headers)
    assert calculated['source_loaded_tensor_bytes']==actual
    assert calculated['source_raw_tensor_bytes']==sum(p.numel()*4 for p in model.parameters())


def test_selected_python_versions_not_current_process_metadata(tmp_path):
    cfg=recipe(tmp_path)
    path=tmp_path/'model'/'config.json'
    config=json.loads(path.read_text())
    config['transformers_version']='4.51.3'
    path.write_text(json.dumps(config))
    hw=profile(tmp_path)
    hw['torch']['packages']['transformers']='4.51.3'
    assert plan_specialist(cfg,hw,workspace_parent=tmp_path)['status']=='READY'
    hw['torch']['packages']['transformers']='4.40.0'
    blocked=plan_specialist(cfg,hw,workspace_parent=tmp_path)
    assert blocked['status']=='BLOCKED'
    assert 'source_config_incompatible_with_selected_python_transformers' in blocked['reasons']


def test_auto_request_and_explicit_recipe_device_precedence(tmp_path):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path,devices=[gpu(2)])
    assert plan_specialist(cfg,hw,workspace_parent=tmp_path)['execution_device']=='cuda:2'
    cfg['execution_device']='cpu'
    assert plan_specialist(cfg,hw,workspace_parent=tmp_path)['execution_device']=='cpu'
    assert plan_specialist(cfg,hw,workspace_parent=tmp_path,requested_device='auto')['execution_device']=='cuda:2'


def test_nonfinite_hardware_returns_json_blocked(tmp_path):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path)
    hw['memory']['available_bytes']=float('nan')
    blocked=plan_specialist(cfg,hw,workspace_parent=tmp_path)
    assert blocked['status']=='BLOCKED'
    json.dumps(blocked,allow_nan=False)


def test_malformed_bounded_header_and_symlink_refused(tmp_path):
    cfg=recipe(tmp_path)
    weight=tmp_path/'model'/'model.safetensors'
    weight.write_bytes(struct.pack('<Q',16*1024**2+1))
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and any('header exceeds' in s for s in p['reasons'])
    weight.unlink()
    weight.symlink_to(tmp_path/'training')
    assert plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)['status']=='BLOCKED'


def test_cli_is_json_metadata_only_and_exclusive_output(tmp_path,capsys):
    from asea.hardware.__main__ import main
    cfg=recipe(tmp_path,bound=False)
    path=tmp_path/'recipe.json'
    path.write_text(json.dumps(cfg))
    output=tmp_path/'plan.json'
    # Missing selected executable keeps the discovery bounded and CPU-only.
    code=main(['plan','--recipe',str(path),'--python','/nonexistent/synthetic-python',
               '--device','cpu','--workspace-parent',str(tmp_path),'--output',str(output)])
    assert code==2
    result=json.loads(capsys.readouterr().out)
    assert result['no_execution'] and result['status']=='BLOCKED'
    assert json.loads(output.read_text())==result
    with pytest.raises(FileExistsError):
        main(['plan','--recipe',str(path),'--python','/nonexistent/synthetic-python','--output',str(output)])


def test_cuda_cannot_bypass_missing_cpu_dtype_primitive(tmp_path):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path,devices=[gpu()])
    hw['torch']['cpu']['bfloat16']=False
    p=plan_specialist(cfg,hw,workspace_parent=tmp_path,requested_device='cuda:0')
    assert p['status']=='BLOCKED'
    assert any('reconstruction_is_cpu_only' in reason for reason in p['reasons'])


def test_missing_runtime_dependency_refuses_ready(tmp_path):
    cfg=recipe(tmp_path)
    hw=profile(tmp_path)
    hw['torch']['packages']['peft']=None
    p=plan_specialist(cfg,hw,workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and p['execution_device'] is None


def test_switch_f32_router_cannot_be_rounded_by_recipe(tmp_path):
    cfg=recipe(tmp_path,switch_config())
    config=switch_config()
    config['router_dtype']='bfloat16'
    overrides={key:'F32' for key in _switch_shapes(config)[0] if '.router.classifier.' in key}
    source=write_checkpoint(tmp_path/'other',config,overrides=overrides)
    cfg['source_path']=str(source)
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and any('round original F32' in r for r in p['reasons'])


@pytest.mark.parametrize('phase', ['source_evaluation', 'inference_evaluation'])
def test_unprofiled_2k_envelope_auto_cpu_matches_runtime_800mib(tmp_path, monkeypatch, phase):
    """Tiny headers; SYNTHETIC CUDA admission only, never GPU execution evidence."""
    from asea.specialist import evaluation as e, devices as d
    from test_specialist_devices import fake_torch
    raw = dict(qwen_config(heads=8), max_position_embeddings=4096)
    cfg = recipe(tmp_path, raw)
    cfg.update(max_length=256, max_new_tokens=256, execution_device='auto')
    saved = copy.deepcopy(cfg)
    hardware = profile(tmp_path, devices=[gpu(free=800*1024**2)])
    plan = plan_specialist(cfg, hardware, workspace_parent=tmp_path, profile_data=False)
    assert plan['status']=='PLANNING_ONLY' and plan['execution_device'] is None, plan['reasons']
    assert plan['estimates']['device_candidates_rejected']=={'cuda:0':'insufficient_cuda_vram'}
    assert 'synthetic' in plan['evidence_level']
    assert cfg==saved  # no smaller training/context/generation/rank bound
    explicit = plan_specialist(cfg, hardware, workspace_parent=tmp_path, requested_device='cuda:0', profile_data=False)
    assert explicit['status']=='BLOCKED' and explicit['execution_device'] is None
    envelope = plan['estimates']['inference_envelopes'][phase]
    assert envelope['prefill_tokens']==2048 and envelope['max_new_tokens']==256
    assert envelope['device_peak_bytes']>800*1024**2-256*1024**2
    generator = e.NativeGenerator.__new__(e.NativeGenerator)
    generator.raw = dict(raw)
    generator.manifest = None
    generator.max_input_tokens = None
    if phase=='inference_evaluation':
        generator.raw['intermediate_size'] = plan['model']['counts']['reduction']['output_intermediate_size']
        generator.manifest = {'counts': {'rank': cfg['recovery']['rank']}}
    generator.dtype, generator.max_new_tokens = cfg['dtype'], cfg['max_new_tokens']
    generator.memory_budget_bytes = generator.device_memory_budget_bytes = None
    generator.torch = fake_torch(free=(800*1024**2,))
    loaded = envelope['loaded_tensor_bytes']
    generator.admission = {'header_loaded_bytes': loaded, 'source_payload_bytes': loaded, 'estimated_additional_peak_bytes': loaded*2+512*1024**2}
    probes = []
    def capability_only(torch, device, dtype, *, operators=True):
        probes.append(operators)
        assert not operators, 'undersized GPU must be refused BEFORE operator execution'
        return {'evidence': 'SYNTHETIC capability only; GPU_UNVERIFIED'}
    monkeypatch.setattr(d, '_cuda_probe', capability_only)
    # Host admission stays real; only memory/capability of CUDA is synthetic.
    for request in ('auto', 'cpu'):
        generator.execution_device_requested = request
        generator._select_execution()
        assert generator.execution_device=='cpu'
        assert generator.inference_envelope==envelope
    generator.execution_device_requested = 'cuda:0'
    with pytest.raises(d.DeviceRejected, match='CUDA memory admission'):
        generator._select_execution()
    assert probes==[False, False]
    short = e._admit_inference(generator.raw, cfg['dtype'], 256, 256, _estimate_only=True)
    assert loaded+128*1024**2+short['estimated_additional_peak_bytes']<800*1024**2-512*1024**2


@pytest.mark.parametrize('family', ['qwen2', 'switch_transformers'])
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('mode', ['native_merged', 'factor_preserving'])
def test_source_recovered_envelopes_match_runtime_storage_and_workspace(tmp_path, monkeypatch, family, dtype, mode):
    from asea.specialist import evaluation as e
    raw = dict(qwen_config(), max_position_embeddings=1024) if family=='qwen2' else switch_config()
    cfg = recipe(tmp_path, raw)
    cfg['dtype'] = dtype
    cfg['recovery']['export_mode'] = mode
    plan = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert plan['status']=='READY', plan['reasons']
    shapes = (_qwen_shapes(raw, .75) if family=='qwen2' else _switch_shapes(raw))[:2]
    # Loader arithmetic only; do not admit/allocate real or fabricated GPU memory.
    monkeypatch.setattr(e, '_runtime_admission', lambda phase, estimate, fields, budget: dict(fields, estimated_additional_peak_bytes=estimate))
    for index, phase in enumerate(('source_evaluation', 'inference_evaluation')):
        headers = {k: {'numel': math.prod(s), 'dtype': 'BF16'} for k,s in shapes[index].items()}
        config = dict(raw)
        rank = 0
        if index:
            if family=='qwen2':
                config['intermediate_size'] = plan['model']['counts']['reduction']['output_intermediate_size']
            else:
                config.update(model_type='t5', architectures=['T5ForConditionalGeneration'])
            if mode=='factor_preserving':
                rank = cfg['recovery']['rank']
                headers['synthetic.lora_A.weight'] = {'numel': plan['model']['counts']['factor_parameters'], 'dtype': 'F32'}
        envelope = plan['estimates']['inference_envelopes'][phase]
        from asea.hardware.metadata import _target_dtype
        from asea.specialist.devices import EXPLICIT_NATIVE_LOADER
        targets = {k: ('float32' if '.lora_' in k or
            _target_dtype(k, config['model_type'], dtype, v['dtype'], config) == 'F32' else 'bfloat16')
            for k, v in headers.items()}
        headers = {k: dict(v, file='arithmetic-only.safetensors') for k, v in headers.items()}
        inventory = {'arithmetic-only.safetensors': {'size': sum(v['numel'] * (4 if v['dtype']=='F32' else 2)
                     for v in headers.values()) + 1024}}
        loader = e._admit_loader(headers, config, dtype,
            load_plan={'strategy': EXPLICIT_NATIVE_LOADER, 'target_dtypes': targets}, inventory=inventory)
        workspace = e._admit_inference(config, dtype, plan['data_profile']['evaluation_input_profile']['max_input_tokens'],
            cfg['max_new_tokens'], factor_rank=rank, _estimate_only=True)
        assert envelope['loaded_tensor_bytes']==loader['header_loaded_bytes']
        assert envelope['workspace']==workspace
        assert envelope['device_peak_bytes']==loader['header_loaded_bytes']+128*1024**2+workspace['estimated_additional_peak_bytes']
        assert plan['estimates']['cuda_phase_bytes'][phase]==envelope['device_peak_bytes']
        assert bool(workspace['factor_workspace_bytes'])==bool(rank)
        if family=='switch_transformers' and dtype=='bfloat16':
            assert loader['header_loaded_bytes']>sum(v['numel']*2 for v in headers.values())


def test_source_tokenizer_presence_required_but_not_loaded(tmp_path):
    cfg=recipe(tmp_path)
    (tmp_path/'model'/'tokenizer.json').unlink()
    p=plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status']=='BLOCKED' and any('tokenizer assets missing' in r for r in p['reasons'])


def test_native_profile_matches_recovery_selection_eos_bank_and_logits(tmp_path, monkeypatch):
    import random
    import transformers
    from asea.specialist.recovery import _read_samples, _encode_details
    from asea.specialist.workflow import load_recipe
    cfg = recipe(tmp_path)
    cfg['seed'] = 319
    cfg['recovery']['max_samples'] = 1
    def no_models(*a, **kw):
        raise AssertionError('preflight may not construct/load models')
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', no_models)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, 'from_pretrained', no_models)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'READY', p['reasons']
    normalized = load_recipe(cfg)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['source_path'], local_files_only=True)
    splits = [[(r, _encode_details(r, tokenizer, 'causal', cfg['max_length'])[0])
               for r in _read_samples(Path(cfg[key]))[0]] for key in ('training', 'validation_data')]
    rng = random.Random(normalized['seed'])
    for split in splits:
        rng.shuffle(split)
    chosen = [split[:1] for split in splits]
    examples = [e for split in chosen for _, e in split]
    counts = [sum(v != -100 for v in e['labels'][1:]) for e in examples]
    length = max(max(len(e['input_ids']), len(e['labels'])) for e in examples)
    for name, split in zip(('training', 'validation'), chosen):
        assert [r['id'] for r, _ in split] == [r['id'] for r in p['data_profile']['selected'][name]]
    e = p['estimates']
    assert e['cached_teacher_bank_max_rows'] == 2
    assert e['cached_teacher_bank_response_tokens'] == sum(counts) == 4  # response + EOS each
    assert e['cached_teacher_bank_disk_bytes'] == sum(counts)*128*4 + 2*16384 + 1024**2
    assert e['training_logits_workspace_bytes'] == length*128*2 + max(counts)*128*(2+24)
    assert p['data_profile']['all_rows_validated'] and p['data_profile']['model_weights_materialized'] is False
    assert 'available_bytes' in p['data_profile']['observed_memory_after_processing']
    for key in ('training', 'validation_data'):
        entry = p['bindings']['data_metadata'][key]
        assert len(entry['sha256']) == 64 and entry['path'] == cfg[key]
        assert entry['file_rows'] == 2 and entry['selected_rows'] == 1


def test_declared_qwen_half_billion_42_plus_8_bank_not_512_rows(tmp_path, monkeypatch):
    """Declared 0.5B dimensions only; real tiny tokenizer, no model or GPU claims."""
    from types import SimpleNamespace
    from asea.hardware.data import read_permitted, native_profile
    from asea.hardware.planning import estimate_phases
    from asea.specialist.workflow import load_recipe
    from asea.specialist import recovery
    cfg = recipe(tmp_path)
    # Distinct synthetic plain TRAIN/DEV text. 48 response words plus EOS.
    train = [dict(id='t%d'%i,family='train-%d'%i,prompt='alpha '+('one '*(i+1)),response='two '*48) for i in range(42)]
    dev = [dict(id='v%d'%i,family='dev-%d'%i,prompt='gamma '+('three '*(i+1)),response='four '*48) for i in range(8)]
    cfg.update(write_permitted_data(tmp_path, train, dev), max_length=256)
    cfg['recovery']['max_samples'] = 256
    cfg = load_recipe(cfg)
    # No large tensors created; this model is explicitly config-only.
    raw = qwen_config(896,4864,24,151936,14,2,True)
    root = write_checkpoint(tmp_path/'declared-0.5b',raw,bound=False)
    from asea.hardware.metadata import inspect_source
    model, config = inspect_source(root,.75,'bfloat16',4)
    assert model['checkpoint_bound'] is False
    splits, metadata = read_permitted(cfg,{})
    native = native_profile(cfg, config, splits)
    native['_metadata'] = metadata
    _, estimates = estimate_phases(model, config, cfg, native)
    # Call runtime _preflight arithmetic without native model/meta construction.
    # Header stats below are synthetic small stand-ins; response/logit/disk bank
    # formulas use the same declared vocabulary and real encoded selected rows.
    stats = dict(adapter_parameters=8,loaded_bytes=1024,largest_lora_target_parameters=16,
                 lora_target_loaded_bytes=64,load_peak_bytes=2048,largest_tensor_parameters=128,
                 tokenizer_asset_bytes=1024,tensor_count=8)
    monkeypatch.setattr(recovery, '_header_stats', lambda *a, **kw:dict(stats))
    monkeypatch.setattr(recovery, '_available_ram', lambda:1024*GiB)
    # Arithmetic-only fixture: disk capacity is synthetic, just like headers/RAM.
    # Do not make this comparison depend on unrelated host artifact usage.
    with monkeypatch.context() as capacity:
        capacity.setattr(recovery.shutil, 'disk_usage',
                         lambda path: SimpleNamespace(total=64*GiB, used=32*GiB, free=32*GiB))
        resources = recovery._preflight(root, root, tmp_path, SimpleNamespace(**raw),4,[],2,
            native['selected_max_complete_length'],True,object(),teacher_mode='cached',
            max_response_tokens=native['max_response_tokens'],total_response_tokens=native['total_response_tokens'],
            bank_rows=native['selected_rows'],export_mode='factor_preserving')
    assert estimates['cached_teacher_bank_max_rows'] == 50
    assert estimates['cached_teacher_bank_disk_bytes'] == resources['bank_disk_bytes_bound']
    assert estimates['training_logits_workspace_bytes'] == resources['logits_workspace_bytes']
    assert 1.3*GiB < estimates['cached_teacher_bank_disk_bytes'] < 1.5*GiB
    assert estimates['workspace_disk_required_bytes'] < 5*GiB*.9
    assert estimates['cached_teacher_bank_disk_bytes'] < 512*256*151936*4/40
    # Config-only plans still cannot turn these measurements into hardware claims.
    cfg['source_path'] = str(root)
    plan = plan_specialist(cfg, profile(tmp_path,ram=1024*GiB,disk=1024*GiB), workspace_parent=tmp_path)
    assert plan['status'] == 'PLANNING_ONLY'


def test_header_only_and_unavailable_tokenizer_never_ready(tmp_path):
    cfg = recipe(tmp_path)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path, profile_data=False)
    assert p['status'] == 'PLANNING_ONLY'
    assert all(not v['contents_read'] for v in p['bindings']['data_metadata'].values())
    (tmp_path/'model'/'tokenizer.json').write_text('{}')
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'PLANNING_ONLY', p['reasons']
    assert p['data_profile']['status'] == 'unavailable'
    assert p['estimates']['cached_teacher_bank_max_rows'] == 4


def test_unselected_oversize_row_cannot_escape_native_validation(tmp_path):
    cfg = recipe(tmp_path)
    train = [dict(id='t1',family='train-a',prompt='alpha',response='one'),
             dict(id='t2',family='train-b',prompt='beta',response='two '*100)]
    cfg.update(write_permitted_data(tmp_path, train=train))
    cfg['recovery']['max_samples'] = 1
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED'
    assert any('Complete token length exceeds' in r for r in p['reasons'])


@pytest.mark.parametrize('kind', ['path_prefix','manifest_usage','alias','malformed_final_path','malformed_path_container','malformed_manifest'])
def test_source_eligibility_no_final_leak_before_any_raw_data_open(tmp_path, monkeypatch, kind):
    cfg = recipe(tmp_path)
    manifest_path = tmp_path/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if kind == 'path_prefix':
        cfg['training'] = str(tmp_path/'source-calibration-only.json')
    elif kind == 'manifest_usage':
        manifest['usage'] = 'source_calibration_only'
    elif kind == 'alias':
        manifest['final_suite_path'] = cfg['training']
    elif kind == 'malformed_final_path':
        manifest['final_suite_path'] = {'path':42}
    elif kind == 'malformed_path_container':
        manifest['paths'] = ['final-suite.json']
    else:
        manifest = []
    manifest_path.write_text(json.dumps(manifest))
    original = Path.open
    forbidden = {Path(cfg['training']),Path(cfg['validation_data']),Path(cfg['calibration']),
                 Path(cfg['validation_suite']),tmp_path/'final-suite.json'}
    def sentinel(path, *a, **kw):
        assert path not in forbidden, 'raw data/final/suite opened before eligibility gate: '+str(path)
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', sentinel)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED', p


def test_source_eligibility_no_final_leak_hardlink_alias(tmp_path, monkeypatch):
    import os
    cfg = recipe(tmp_path)
    final = tmp_path/'final-suite.json'
    final.unlink()
    os.link(cfg['training'], final)
    original = Path.open
    def sentinel(path, *a, **kw):
        assert path not in {Path(cfg['training']), final}
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', sentinel)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED' and any('aliases a final' in r for r in p['reasons'])


def test_permitted_profile_never_opens_final_suite_or_calibration(tmp_path, monkeypatch):
    cfg = recipe(tmp_path)
    original = Path.open
    forbidden = {Path(cfg['calibration']),tmp_path/'final-suite.json'}
    def sentinel(path, *a, **kw):
        assert path not in forbidden, 'preflight read an unpermitted raw file'
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', sentinel)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'READY', p['reasons']
    assert p['data_profile']['_metadata']['final_opened'] is False
    assert not any('final' in Path(path).name for path in p['data_profile']['_metadata']['hashes'])


@pytest.mark.parametrize('changed', ['train.json','manifest.json','selection-lock.json','validation-suite.json','model/tokenizer.json'])
def test_input_sha_rechecked_after_native_processing(tmp_path, monkeypatch, changed):
    from asea.hardware import data
    cfg = recipe(tmp_path)
    original = data.profile_native_selected
    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        with (tmp_path/changed).open('a') as handle:
            handle.write(' ')
        return value
    monkeypatch.setattr(data, 'profile_native_selected', mutate)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED' and any('changed after reading' in r for r in p['reasons'])


def test_input_sha_rechecked_around_row_validation(tmp_path, monkeypatch):
    from asea.hardware import data
    cfg = recipe(tmp_path)
    original = data._read_samples
    def mutate(path):
        rows = original(path)
        assert isinstance(path, bytes)  # validator consumes the admitted snapshot
        with Path(cfg['training']).open('a') as handle:
            handle.write(' ')
        return rows
    monkeypatch.setattr(data, '_read_samples', mutate)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED' and any('changed after reading' in r for r in p['reasons'])


def test_selected_python_native_tokenizer_worker_and_missing_python(tmp_path):
    cfg = recipe(tmp_path)
    # Different path spelling forces fixed isolated worker; never a model run.
    alternate = str(Path(sys.executable).parent/'..'/'bin'/Path(sys.executable).name)
    p = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path, python_executable=alternate)
    assert p['status'] == 'READY', p['reasons']
    assert p['data_profile']['python_executable'] == str(Path(sys.executable).resolve())
    missing = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path, python_executable='/no/such/python')
    assert missing['status'] == 'PLANNING_ONLY'
    assert missing['data_profile']['status'] == 'unavailable'


def test_device_cap_subtracts_reserved_not_allocated_or_driver_free(tmp_path):
    cfg = recipe(tmp_path)
    hw = profile(tmp_path, devices=[gpu()])
    hw['torch']['devices'][0].update(allocated_bytes=GiB,reserved_bytes=2*GiB)
    initial = plan_specialist(cfg, hw, workspace_parent=tmp_path, requested_device='cuda:0')
    peak = initial['estimates']['cuda_peak_bytes']
    cfg['device_memory_budget_bytes'] = peak+2*GiB
    fit = plan_specialist(cfg, hw, workspace_parent=tmp_path, requested_device='cuda:0')
    assert fit['status'] == 'READY', fit['reasons']
    budget = fit['estimates']['cuda_budget']
    assert budget['effective_bytes'] == peak
    assert budget['operator_remaining_bytes'] == peak and budget['reserved_bytes'] == 2*GiB
    assert budget['reclaimed_memory_credit_bytes'] == 0
    cfg['device_memory_budget_bytes'] -= 1
    blocked = plan_specialist(cfg, hw, workspace_parent=tmp_path, requested_device='cuda:0')
    assert blocked['status'] == 'BLOCKED'
    assert blocked['estimates']['cuda_candidate_budgets']['cuda:0']['effective_bytes'] == peak-1
    auto = plan_specialist(cfg, hw, workspace_parent=tmp_path, requested_device='auto')
    assert auto['status'] == 'READY' and auto['execution_device'] == 'cpu'
    # GPU ceiling must not turn into an accidental host budget.
    cfg['device_memory_budget_bytes'] = 1
    cpu = plan_specialist(cfg, hw, workspace_parent=tmp_path, requested_device='cpu')
    assert cpu['status'] == 'READY' and cpu['estimates']['host_budget']['operator_cap_bytes'] is None


@pytest.mark.parametrize('cap', [True,False,0,-1,1.5,'1024',{},[]])
def test_device_cap_strict_schema(tmp_path, cap):
    cfg = recipe(tmp_path)
    cfg['device_memory_budget_bytes'] = cap
    assert plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)['status'] == 'BLOCKED'


@pytest.mark.parametrize('allocated,reserved', [(None,0),(0,None),(-1,0),(2,1),(False,0),(0,50*GiB)])
def test_device_cap_requires_valid_selected_allocator_observations(tmp_path, allocated, reserved):
    cfg = recipe(tmp_path)
    hw = profile(tmp_path,devices=[gpu()])
    hw['torch']['devices'][0].update(allocated_bytes=allocated,reserved_bytes=reserved)
    p = plan_specialist(cfg,hw,workspace_parent=tmp_path,requested_device='cuda:0')
    assert p['status'] == 'BLOCKED'
    assert any('allocated_reserved' in r for r in p['reasons'])


def test_torch_probe_records_each_device_allocator_after_primitive_probes(monkeypatch, capsys):
    """Fake primitive mechanics only, not execution or real hardware evidence."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    from asea.hardware.probe import _TORCH_SCRIPT
    calls = []
    class Scalar:
        def __matmul__(self, other): return self
        def all(self): return self
        def item(self): return True
        def __bool__(self): return True
    class Cuda:
        current = None
        @contextmanager
        def device(self, index):
            self.current = index
            yield
            self.current = None
        def is_available(self): return True
        def device_count(self): return 2
        def get_device_properties(self, index):
            return SimpleNamespace(name='SYNTHETIC',total_memory=8*GiB,major=8,minor=0)
        def is_bf16_supported(self, **kwargs): return True
        def synchronize(self, index): calls.append(('primitive',index))
        def mem_get_info(self, index): return 6*GiB,8*GiB
        def memory_allocated(self, index):
            assert index == self.current
            calls.append(('allocated',index))
            return (index+1)*1024
        def memory_reserved(self, index):
            assert index == self.current
            calls.append(('reserved',index))
            return (index+1)*2048
    fake = SimpleNamespace(__version__='FAKE',version=SimpleNamespace(cuda='FAKE',hip=None),
        cuda=Cuda(),backends=SimpleNamespace(),set_num_threads=lambda n:None,
        float32='float32',bfloat16='bfloat16',ones=lambda *a,**kw:Scalar(),isfinite=lambda x:x)
    monkeypatch.setitem(sys.modules,'torch',fake)
    exec(_TORCH_SCRIPT,{})
    report = json.loads(capsys.readouterr().out.split('SILT_HARDWARE_JSON=')[1])
    for index, device in enumerate(report['devices']):
        assert device['probe_valid']
        assert device['allocated_bytes'] == (index+1)*1024
        assert device['reserved_bytes'] == (index+1)*2048
        assert calls.index(('allocated',index)) > calls.index(('primitive',index))


def test_live_profile_charges_post_tokenizer_observed_ram(tmp_path, monkeypatch):
    from asea.hardware import data
    cfg = recipe(tmp_path)
    original = data.profile_native_selected
    def lowered(*args, **kwargs):
        value = original(*args, **kwargs)
        value['observed_memory_after_processing']['available_bytes'] = 128*1024**2
        return value
    monkeypatch.setattr(data,'profile_native_selected',lowered)
    hw = profile(tmp_path)
    hw['profile_kind'] = 'detected'  # fabricated live-snapshot branch test, NOT hardware evidence
    p = plan_specialist(cfg,hw,workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED'
    assert p['hardware']['memory']['available_bytes'] == 128*1024**2
    assert any('insufficient_host_ram' in r for r in p['reasons'])


def test_native_tokenizer_asset_set_pinned_before_after(tmp_path, monkeypatch):
    from asea.hardware import data
    cfg = recipe(tmp_path)
    original = data.profile_native_selected
    def new_template(*args, **kwargs):
        value = original(*args, **kwargs)
        (tmp_path/'model'/'chat_template.jinja').write_text('changed template')
        return value
    monkeypatch.setattr(data,'profile_native_selected',new_template)
    p = plan_specialist(cfg,profile(tmp_path),workspace_parent=tmp_path)
    assert p['status'] == 'BLOCKED' and any('asset set changed' in r for r in p['reasons'])

