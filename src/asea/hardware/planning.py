"""Capability/data preflight; never materialize models, train, approve or reserve."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

from asea.artifacts import digest, safe_path
from .metadata import inspect_source
from .probe import probe_hardware, disk_snapshot

MiB = 1024**2
GiB = 1024**3


def _recipe(value):
    from asea.specialist.workflow import load_recipe
    from asea.specialist.recovery import _bounded_json
    from asea.artifacts import safe_file
    # Preserve whether device selection was explicit; the legacy workflow loader
    # defaults execution to CPU, whereas this standalone planner defaults to auto.
    if isinstance(value, dict):
        raw, cfg = value, load_recipe(value)
    else:
        from .data import fingerprint
        path = safe_file(value)
        before = fingerprint(path)
        try:
            raw = _bounded_json(path)
            cfg = load_recipe(value)
        finally:
            if fingerprint(path) != before:
                raise ValueError('recipe changed during bounded planning read')
    return cfg, raw.get("execution_device", "auto")


def _budget(available, cap=None, gpu=False):
    if type(available) is not int or available <= 0:
        return {'available_bytes': available, 'reserve_bytes': None, 'operator_cap_bytes': cap, 'effective_bytes': 0}
    reserve = max(256*MiB, available//10)
    usable = max(0, available-reserve)
    if cap is not None:
        usable = min(usable, cap)
    return {'available_bytes': available, 'reserve_bytes': reserve, 'operator_cap_bytes': cap, 'effective_bytes': usable}


def estimate_phases(model, config, recipe, data_profile=None, *, inference_scope='dev'):
    """Upper estimates are explicit conservative assumptions, not RSS predictions.

    Lower bounds count unique native storage. Full F32 logits, eager attention,
    tokenizer Python objects, native casts and export verification stay separate.
    """
    c = model['counts']
    h = config.get('hidden_size',config.get('d_model'))
    width = config.get('intermediate_size',config.get('d_ff'))
    layers = config.get('num_hidden_layers',config.get('num_layers'))
    dec = config.get('num_decoder_layers') or layers
    heads = config.get('num_attention_heads',config.get('num_heads'))
    vocab = config['vocab_size']
    configured_length, generated = recipe['max_length'], recipe['max_new_tokens']
    prof = data_profile or {}
    verified = prof.get('status') == 'verified'
    length = prof['selected_max_complete_length'] if verified else configured_length
    response = prof['max_response_tokens'] if verified else length
    byte = 2 if recipe['dtype']=='bfloat16' else 4
    r = recipe['recovery']
    kd = r['kd_weight']>0
    cached = r['teacher_mode']=='cached' and kd
    factor = r['export_mode']=='factor_preserving'
    source, base = c['source_loaded_tensor_bytes'], c['base_tensor_bytes']
    runtime = 512*MiB
    tokenizer = max(64*MiB, model['tokenizer_asset_bytes']*8)
    mapping = max(0, model['tensor_file_bytes']-c['source_raw_tensor_bytes']) + c['source_tensor_count']*8192
    source_load = c['source_load_peak_lower_bound_bytes'] + mapping
    base_mapping = c['base_tensor_count']*10240 + 16*MiB
    base_load = base + base_mapping + c['student_cast_source_pages_bytes'] + c['student_cast_scratch_bytes']
    # Native Switch source forwards may upcast routers; original F32 is preserved.
    replacements = 0
    if c['model_type']=='qwen2':
        replacements = layers*h*c['reduction']['output_intermediate_size']*3*byte
    # Calibration is not opened by this planner; reconstruction keeps its full bound.
    forward = configured_length*(vocab*8+width*32+h*32)+configured_length**2*heads*16
    layer_cast = h*width*12
    serialization = max(min(source,256*MiB),c['source_largest_tensor_bytes'])
    reconstruction = source_load+replacements+max(forward,layer_cast,serialization)+runtime+tokenizer
    activations = max(128*MiB,length*h*(layers+dec if c['model_type']!='qwen2' else layers)*byte*12)
    # Include full attention matrix (eager) in addition to layer-state factor.
    activations += length**2*heads*(layers+dec if c['model_type']!='qwen2' else layers)*4
    # Identical response-position/F32 bank arithmetic to recovery._preflight.
    train_logits = length*vocab*byte+response*vocab*(byte+(24 if kd else 20))
    teacher_logits = vocab*(length*max(byte,4)+response*(byte+4))
    teacher_work = teacher_logits+activations
    teacher = source_load+teacher_work+runtime+tokenizer
    adapters = c['factor_optimizer_gradient_bytes']+c['factor_snapshot_bytes']
    merge = 0 if factor else c['largest_lora_target_parameters']*8+c['lora_target_tensor_bytes']
    adaptation = base_load+adapters+train_logits+activations+merge+runtime+tokenizer
    if kd and not cached:
        adaptation += source_load+teacher_work
    probe = activations+6*(length+2)*vocab*max(byte,4)+(layers+dec)*(length+2)*h*byte*4
    export_io = 64*MiB+16*MiB+tokenizer+c['base_tensor_count']*8192
    if factor:
        export = base_load+adapters+probe+export_io+runtime
    else:
        export = base*2+max(activations+train_logits,c['base_largest_tensor_parameters']*8)+adapters+runtime+tokenizer
    # Bind both DEV evaluations to actual complete rendered suite input lengths.
    # Recovery training lengths are not inference lengths. No quality knob changes.
    from asea.specialist.devices import inference_envelope, EXPLICIT_NATIVE_LOADER, NATIVE_BUFFER_RESERVE
    # Version compatibility belongs to the selected interpreter's hardware probe,
    # not this process's package installation. Structural preflight already ran.
    source_config = {k:v for k,v in config.items() if k!='transformers_version'}
    recovered_config = dict(source_config)
    if c['model_type']=='qwen2':
        recovered_config['intermediate_size'] = c['reduction']['output_intermediate_size']
    else:
        recovered_config.update(model_type='t5', architectures=['T5ForConditionalGeneration'])
    source_infer_loaded = source  # architecture/header-derived native dtype policy, not a family-wide F32 guess
    recovered_infer_loaded = base + (c['factor_tensor_bytes'] if factor else 0)
    if inference_scope not in ('dev', 'final'):
        raise ValueError('inference_scope must be dev or final')
    input_bound = prof.get('evaluation_input_profile', {}).get('max_input_tokens') if verified and inference_scope == 'dev' else None
    if verified and inference_scope == 'dev' and input_bound is None:
        raise ValueError('verified DEV plan requires complete reviewed suite input profile')
    source_envelope = inference_envelope(source_config, recipe['dtype'], generated, source_infer_loaded,
        max_input_tokens=input_bound)
    recovered_envelope = inference_envelope(recovered_config, recipe['dtype'], generated,
        recovered_infer_loaded, factor_rank=r['rank'] if factor else 0, max_input_tokens=input_bound)
    source_work = source_envelope['workspace']['estimated_additional_peak_bytes']
    recovered_work = recovered_envelope['workspace']['estimated_additional_peak_bytes']
    # Loader mmap/cast transients and inference do not coexist. Count the same
    # native loaded storage ONCE, plus the maximum of serial phase workspaces.
    from asea.specialist.devices import inference_host_phases
    source_host = inference_host_phases(source_infer_loaded, c['source_raw_tensor_bytes'], source_work,
        tokenizer_bytes=tokenizer, retained_cast_source_bytes=c['source_cast_pages_bytes'],
        loader_strategy=EXPLICIT_NATIVE_LOADER, largest_cast_scratch_bytes=c['source_cast_scratch_bytes'],
        mapping_overhead_bytes=mapping, buffer_reserve_bytes=NATIVE_BUFFER_RESERVE)
    recovered_host = inference_host_phases(recovered_infer_loaded, recovered_infer_loaded, recovered_work,
        tokenizer_bytes=tokenizer, retained_cast_source_bytes=c['student_cast_source_pages_bytes'],
        loader_strategy=EXPLICIT_NATIVE_LOADER, largest_cast_scratch_bytes=c['student_cast_scratch_bytes'],
        mapping_overhead_bytes=base_mapping, buffer_reserve_bytes=NATIVE_BUFFER_RESERVE,
        extra_loader_bytes=2*c['factor_tensor_bytes'] if factor else 0)
    source_evaluate = max(source_host['estimated_additional_peak_bytes'], source_load+runtime+tokenizer)
    inference = max(recovered_host['estimated_additional_peak_bytes'], base_load+runtime+tokenizer)
    # Final inputs remain unconsumed. This separate, unprofiled conservative
    # envelope is NOT evidence of final fit and does not veto a DEV-only build.
    final_envelopes = {
        'source': inference_envelope(source_config, recipe['dtype'], generated, source_infer_loaded),
        'recovered': inference_envelope(recovered_config, recipe['dtype'], generated,
            recovered_infer_loaded, factor_rank=r['rank'] if factor else 0)}
    kv_cache = max(source_envelope['workspace']['kv_cache_bytes'], recovered_envelope['workspace']['kv_cache_bytes'])
    # Complete TRAIN/DEV validation precedes the exact native seeded selection.
    # Row-only fallback remains conservative and never grants READY.
    selected_rows = prof.get('selected_rows', 2*r['max_samples'])
    bank_rows = selected_rows if cached else 0
    bank_tokens = (prof['total_response_tokens'] if verified else bank_rows*length) if cached else 0
    bank = bank_tokens*vocab*4+bank_rows*16384+MiB if cached else 0
    output_lower = base+(c['factor_tensor_bytes'] if factor else 0)
    # Header and metadata bounds are explicit, not file bytes confused with RSS.
    output_required = output_lower+model['tokenizer_asset_bytes']+64*MiB+c['base_tensor_count']*2048
    # One retained reconstructed control plus recovery's own two-base scratch/
    # export admission; do not charge the exported base a fourth time through
    # output_required. Existing source assets are already on disk, not copied.
    reconstruction_disk = base+model['tokenizer_asset_bytes']+64*MiB+c['base_tensor_count']*2048
    recovery_disk = base*2+(c['factor_tensor_bytes']*2 if factor else 0)+bank+64*MiB
    if factor:
        recovery_disk += model['tokenizer_asset_bytes']+c['base_tensor_count']*2048
    disk = reconstruction_disk+recovery_disk+128*MiB
    phase_bytes = {'source_evaluation':source_evaluate,'reconstruction':reconstruction,
                   'teacher_cache':teacher if cached else 0,'adaptation':adaptation,'export':export,
                   'inference_evaluation':inference}
    if inference_scope == 'final':
        phase_bytes = {k:v for k,v in phase_bytes.items() if k in ('source_evaluation', 'inference_evaluation')}
        disk = 128*MiB  # bounded final reports only; no reconstruction/training/cache export
    # CPU loaders remain in the CUDA pipeline. Host reconstruction cannot be
    # bypassed with spare VRAM. Current device path retains CPU admission checks.
    phases = [{'name':name,'execution_device':'cpu' if name=='reconstruction' else None,
               'host_ram_bytes':value,'device_ram_bytes':0,'estimate_kind':'conservative_upper_estimate'}
              for name,value in phase_bytes.items()]
    # Recovery's CUDA guards include native eager-attention/backward work. Use
    # the very same native workspace estimator and selected complete lengths;
    # keep source/recovered evaluation's separately profiled envelope above.
    from asea.specialist.devices import recovery_workspace
    def recovery_cuda_work(raw, training=False):
        return recovery_workspace(raw, recipe['dtype'], length, response, training=training,
            factor_rank=r['rank'] if training else 0,
            factor_parameters=c['factor_tensor_bytes']//4 if training else 0, kd=kd)
    teacher_lifetime = recovery_cuda_work(source_config) if kd else None
    student_lifetime = recovery_cuda_work(recovered_config, True)
    cuda_teacher_work = teacher_lifetime['estimated_additional_peak_bytes'] if kd else 0
    cuda_student_work = student_lifetime['estimated_additional_peak_bytes']+merge
    cuda_probe_work = max(probe+export_io+c['factor_optimizer_gradient_bytes'], recovery_cuda_work(recovered_config)['estimated_additional_peak_bytes'])
    resident = source+cuda_teacher_work if kd and not cached else 0
    gpu_values = {'source_evaluation':source_envelope['device_peak_bytes'],
                  'teacher_cache':source+cuda_teacher_work+128*MiB if cached else 0,
                  'adaptation':base+cuda_student_work+resident+128*MiB,
                  'export':max(export-runtime-tokenizer, base+cuda_probe_work+merge,
                               base+cuda_probe_work+c['factor_optimizer_gradient_bytes'])+128*MiB,
                  'inference_evaluation':recovered_envelope['device_peak_bytes']}
    if inference_scope == 'final':
        gpu_values = {k:v for k,v in gpu_values.items() if k in phase_bytes}
    return phases, {'source_unique_tensor_bytes':source,'source_raw_tensor_bytes':c['source_raw_tensor_bytes'],
        'base_tensor_bytes':base,'factor_tensor_bytes':c['factor_tensor_bytes'],
        'output_tensor_lower_bound_bytes':output_lower,'output_required_bytes':output_required,
        'proposed_output_budget_bytes':output_required,'workspace_disk_required_bytes':disk,
        'workspace_disk_components':{'retained_reconstructed_control_bytes':reconstruction_disk,
            'recovery_export_and_scratch_bytes':recovery_disk,'workflow_metadata_allowance_bytes':128*MiB},
        'cached_teacher_bank_disk_bytes':bank,'cached_teacher_bank_max_rows':bank_rows,
        'cached_teacher_bank_response_tokens':bank_tokens,'cached_teacher_bank_tensor_bytes':bank_tokens*vocab*4,
        'training_max_complete_length':length,'training_max_response_tokens':response,
        'training_lengths_kind':'native_selected_complete_lengths' if verified else 'configured_conservative_bound',
        'source_cast_pages_bytes':c['source_cast_pages_bytes'],'source_cast_scratch_bytes':c['source_cast_scratch_bytes'],
        'tokenizer_runtime_bytes':tokenizer,'runtime_allowance_bytes':runtime,'activation_bytes':activations,
        'training_logits_workspace_bytes':train_logits,'kv_cache_bytes':kv_cache,
        'inference_envelopes':{'source_evaluation':source_envelope,'inference_evaluation':recovered_envelope},
        'inference_host_phases':{'source_evaluation':source_host,'inference_evaluation':recovered_host},
        'cuda_training_lifetime':student_lifetime, 'cuda_teacher_lifetime':teacher_lifetime,
        'final_evaluation':{'status':'UNPROFILED_NOT_ADMITTED', 'input_consumption':False,
            'envelopes':final_envelopes, 'requires':'separate pre-consumption conservative admission or reviewed fixed input envelope'},
        'cpu_peak_bytes':max(phase_bytes.values()),'cuda_peak_bytes':max(gpu_values.values()),
        'cuda_phase_bytes':gpu_values,'teacher_mode':r['teacher_mode'],'export_mode':r['export_mode'],
        'assumptions':['batch size one; recipe context/generation bounds unchanged',
            'DEV evaluation binds complete reviewed suite input profile; final stays separate conservative unprofiled envelope',
            ('complete train/dev native token lengths validated before seeded selection; runtime must revalidate, never truncate'
             if verified else 'native data encoding unproven; configured token upper bound only, no READY'),
            ('cached bank uses exact selected response positions including EOS, matching recovery._preflight'
             if verified else 'cached bank uses validated row counts when available, otherwise 2*max_samples, times max_length'),
            'no final reads; only governed DEV suite input text tokenized; calibration stays configured; runtime rechecks',
            'native runtime 512MiB; tokenizer max(64MiB,8*asset bytes); eager attention and full F32 logits',
            'verified explicit per-tensor native loader only: all cast source pages, largest cast scratch, mapped metadata and bounded buffers charged; no release credit',
            'CUDA retains conservative CPU admission for the current CPU loaders/reconstruction',
            'output proposal is explicit only; operator budgets and quality knobs are never changed']}


def _device_reason(device, torch, dtype):
    if torch.get('status')!='ok':
        return 'selected_python_torch_probe_unavailable'
    if any(not torch.get('packages',{}).get(p) for p in ('torch','transformers','peft','accelerate','safetensors')):
        return 'selected_python_runtime_dependencies_unavailable'
    if torch.get('cpu',{}).get(dtype) is not True:
        return 'cpu_dtype_primitive_not_verified_reconstruction_is_cpu_only'
    if device=='cpu':
        return None
    match = re.fullmatch(r'cuda:(0|[1-9][0-9]*)',device or '')
    if not match:
        return 'unsupported_explicit_device_no_fallback'
    index = int(match.group(1))
    found = [v for v in torch.get('devices',[]) if v.get('index')==index]
    if len(found)!=1:
        return 'requested_cuda_device_absent_or_ambiguous'
    gpu = found[0]
    if torch.get('hip_version') or gpu.get('vendor')!='NVIDIA' or not torch.get('cuda_version'):
        return 'only_single_nvidia_cuda_execution_supported'
    if gpu.get('probe_valid') is not True or gpu.get('primitives',{}).get(dtype) is not True:
        return 'cuda_dtype_primitive_not_verified'
    if dtype=='bfloat16' and gpu.get('bf16_supported') is not True:
        return 'cuda_native_bf16_unsupported'
    if type(gpu.get('free_bytes')) is not int or type(gpu.get('total_bytes')) is not int or not 0<=gpu['free_bytes']<=gpu['total_bytes']:
        return 'cuda_memory_unknown_or_invalid'
    allocated, reserved = gpu.get('allocated_bytes'), gpu.get('reserved_bytes')
    if type(allocated) is not int or type(reserved) is not int or not 0 <= allocated <= reserved <= gpu['total_bytes']:
        return 'cuda_allocated_reserved_memory_unknown_or_invalid'
    return None


def _cuda_budget(gpu, cap):
    # Operator cap limits total Torch RESERVED, not merely this proposed phase.
    budget = _budget(gpu['free_bytes'], gpu=True)
    remaining = None if cap is None else max(0, cap-gpu['reserved_bytes'])
    if remaining is not None:
        budget['effective_bytes'] = min(budget['effective_bytes'], remaining)
    budget.update(operator_cap_bytes=cap, operator_remaining_bytes=remaining,
        allocated_bytes=gpu['allocated_bytes'], reserved_bytes=gpu['reserved_bytes'],
        operator_cap_semantics='total_selected_device_torch_reserved_plus_additional',
        reclaimed_memory_credit_bytes=0, runtime_recheck_required=True)
    return budget


def plan_specialist(recipe, hardware=None, *, workspace_parent=None, requested_device=None,
                    python_executable=None, profile_data=True, inference_scope='dev'):
    """JSON planning only. READY means estimated feasibility, not authorization.

    Dict/path recipes use the strict workflow schema. By default a bound source
    enables bounded TRAIN/DEV native-tokenizer preflight (not header-only).
    no_execution means no model/training execution; libraries and permitted data
    may be read. Final artifacts are never opened, hashed or tokenized; governed DEV inputs may be.
    profile_data=False preserves header-only PLANNING_ONLY; it cannot grant READY.
    inference_scope='dev' binds complete governed DEV input profiles. 'final'
    uses only conservative unprofiled inference bounds and report-writing disk;
    it NEVER consumes final inputs or claims actual final input fit.
    Caller-supplied profiles are observations, not authenticated hardware proof.
    """
    result = {'schema_version':1,'status':'BLOCKED','execution_device':None,'reasons':[],
              'model':None,'hardware':None,'phases':[],'estimates':{},'bindings':{},
              'data_profile':{'status':'unproven','kind':'header_only','_metadata':{},'read_attempted_paths':[]},
              'no_execution':True,'quality_approval':False,'reservation':False,'evidence_level':'not_evaluated',
              'no_execution_semantics':'no model construction, weight materialization, generation or training; permitted train/dev tokenization may run',
              'runtime_guards_authoritative':True}
    reasons = result['reasons']
    try:
        cfg, recipe_device = _recipe(recipe)
        if type(profile_data) is not bool:
            raise ValueError('profile_data must be boolean')
        if inference_scope not in ('dev', 'final'):
            raise ValueError('inference_scope must be dev or final')
        from .data import permitted_paths, read_permitted, profile_native_selected, check_pins, check_asset_pins
        paths = permitted_paths(cfg)
        parent = safe_path(workspace_parent if workspace_parent is not None else Path.cwd())
        requested = requested_device if requested_device is not None else recipe_device
        if requested is None:
            requested = 'auto'
        if not isinstance(requested,str):
            raise ValueError('requested_device must be cpu, auto or cuda:N')
        result['bindings'] = {'recipe_sha256':digest(cfg),'workspace_parent':str(parent),'requested_device':requested,
            'inference_scope':inference_scope}
        from .data import admit_source_assets
        admit_source_assets(cfg['source_path'], (Path(cfg['data_manifest']).parent / 'final-suite.json', Path(cfg['data_manifest']).parent / 'final.json'))
        model, config = inspect_source(cfg['source_path'],cfg['reconstruction']['retention'],cfg['dtype'],cfg['recovery']['rank'])
        result['model'] = model
        result['bindings']['inventory_sha256'] = model['inventory_sha256']
        if cfg['reconstruction']['family'] not in ('auto',config['model_type']):
            reasons.append('recipe_family_disagrees_with_actual_config')
        data = {}
        for name in ('training','calibration','validation_data','validation_suite','data_manifest','selection_lock'):
            path = paths[name]
            exists = path.is_file()
            data[name] = {'path':str(path),'exists':exists,'bytes':path.stat().st_size if exists else None,'contents_read':False}
            if not exists:
                reasons.append('missing_local_' + name)
        result['bindings']['data_metadata'] = data
        if model['checkpoint_bound'] and profile_data and not reasons:
            read_audit = result['data_profile']['read_attempted_paths']
            try:
                splits, metadata = read_permitted(cfg, model['files'], read_audit)
            finally:
                for entry in data.values():
                    if entry['path'] in read_audit:
                        entry['read_attempted'] = True
                        entry['contents_read'] = None  # may have failed during bounded parsing
            result['data_profile']['_metadata'] = metadata
            for name in ('training', 'validation_data', 'data_manifest', 'selection_lock', 'validation_suite'):
                data[name].update(contents_read=True, **metadata['hashes'][str(paths[name])])
            # Source inspection already pins these assets; verify before and after
            # native loading, and do not rehash/read any checkpoint weight here.
            # Include optional native chat-template files and inventory-local
            # tokenizer references, not just the common tokenizer.json basename.
            token_pins = {str(Path(cfg['source_path'])/name): entry for name, entry in model['files'].items()
                          if not name.endswith('.safetensors')}
            if any(entry['size'] > 64*MiB for entry in token_pins.values()):
                raise ValueError('native tokenizer/source metadata asset exceeds 64 MiB bound')
            from .data import snapshot
            for path, entry in list(token_pins.items()):
                _, guarded = snapshot(path, 64*MiB, forbidden=metadata['final_paths_metadata_only'], single_link=True)
                if any(guarded[k] != entry[k] for k in ('sha256', 'size')):
                    raise ValueError('source tokenizer asset changed since inventory')
                token_pins[path] = guarded
            check_asset_pins(cfg['source_path'], token_pins)
            selected_python = python_executable or (hardware or {}).get('python_executable')
            try:
                native = profile_native_selected(cfg, config, splits, selected_python)
            finally:
                check_pins(metadata['hashes'])
                check_asset_pins(cfg['source_path'], token_pins)
            native['_metadata'] = metadata
            native['read_attempted_paths'] = read_audit
            native['tokenizer_file_pins'] = token_pins
            # Row counts are proven even when native tokenizer dependencies are absent.
            native.setdefault('selected_rows', sum(min(len(s), cfg['recovery']['max_samples']) for s in splits[:2]))
            for name, split, selected_name in (('training', 'train', 'training'), ('validation_data', 'validation', 'validation')):
                data[name].update(file_rows=metadata['counts'][split], ids=metadata['ids'][split],
                                  families=metadata['families'][split])
                if native.get('status') == 'verified':
                    data[name]['selected_ids'] = [r['id'] for r in native['selected'][selected_name]]
                    data[name]['selected_rows'] = len(data[name]['selected_ids'])
            result['data_profile'] = native
            del splits
        snapshot = copy.deepcopy(hardware) if hardware is not None else probe_hardware(python_executable, paths=(cfg['source_path'],str(parent)))
        if not isinstance(snapshot,dict):
            raise ValueError('hardware must be a JSON snapshot object')
        snapshot.setdefault('profile_kind','supplied_unverified')
        # Heavy native imports/tokenization may reduce RAM after the first probe.
        # Live plans charge the least observed remainder, never assumed teardown.
        observed = result['data_profile'].get('observed_memory_after_processing', {}).get('available_bytes')
        if snapshot.get('profile_kind') == 'detected' and type(observed) is int:
            current = snapshot.get('memory', {}).get('available_bytes')
            if type(current) is int:
                snapshot['memory']['available_bytes'] = min(current, observed)
                snapshot['memory']['post_tokenization_available_bytes'] = observed
        result['bindings']['hardware_sha256'] = digest(snapshot)
        result['hardware'] = snapshot
        if snapshot.get('schema_version')!=1:
            reasons.append('unsupported_hardware_snapshot_schema')
        if snapshot.get('platform',{}).get('system')!='Linux':
            reasons.append('execution_platform_unsupported_linux_only')
        cores = snapshot.get('cpu',{}).get('effective_cores')
        if type(cores) not in (int,float) or not math.isfinite(cores) or cores<=0:
            reasons.append('cpu_capacity_unknown_or_zero')
        phases, estimates = estimate_phases(model,config,cfg,result['data_profile'], inference_scope=inference_scope)
        result['inference_scope'] = inference_scope
        result['final_input_fit_proven'] = False
        if inference_scope == 'final':
            estimates['final_evaluation']['status'] = 'CONSERVATIVE_UNPROFILED_ESTIMATE'
        result['phases'],result['estimates'] = phases, estimates
        host = _budget(snapshot.get('memory',{}).get('available_bytes'),cfg['memory_budget_bytes'])
        estimates['host_budget'] = host
        # Recovery/reconstruction's unchanged operator cap bounds additional
        # allocations. NativeGenerator's cap instead includes current process RSS.
        # Reuse the selected tokenizer worker's observed baseline, never pretend
        # it is free memory or subtract it from observed free RAM a second time.
        baseline = result['data_profile'].get('profile_process_rss_bytes')
        evaluation_peak = max(source['host_ram_bytes'] for source in phases
            if source['name'] in ('source_evaluation', 'inference_evaluation'))
        cap = cfg['memory_budget_bytes']
        evaluation_remaining = None if cap is None or type(baseline) is not int else max(0, cap-baseline)
        estimates['evaluation_host_operator_budget'] = {
            'semantics':'TOTAL process cap minus profiled interpreter RSS; runtime reobserves',
            'profiled_process_rss_bytes':baseline, 'operator_remaining_bytes':evaluation_remaining,
            'estimated_additional_peak_bytes':evaluation_peak,
            'estimated_total_process_peak_bytes':None if type(baseline) is not int else baseline+evaluation_peak}
        estimates['host_budget']['semantics'] = 'additional phase bytes versus observed remaining RAM; current RSS already charged to availability'
        if evaluation_remaining is not None and evaluation_peak > evaluation_remaining:
            reasons.append('insufficient_evaluation_total_process_cap: additional %d + profiled RSS %d > TOTAL operator cap %d' % (
                evaluation_peak, baseline, cap))
        if estimates['cpu_peak_bytes']>host['effective_bytes']:
            reasons.append('insufficient_host_ram: estimated %d > effective %d (%s)' % (
                estimates['cpu_peak_bytes'],host['effective_bytes'],
                'conservative unprofiled FINAL envelope; no final input consumed' if inference_scope == 'final' else
                'GPU cannot replace CPU loader/reconstruction RAM'))
        if inference_scope != 'final' and estimates['output_required_bytes']>cfg['output_budget_bytes']:
            reasons.append('output_budget_exceeded: base=%d factors=%d conservative_output=%d > operator_cap=%d; proposed budget is not applied' % (
                estimates['base_tensor_bytes'],estimates['factor_tensor_bytes'],estimates['output_required_bytes'],cfg['output_budget_bytes']))
        estimates['operator_output_budget_bytes'] = cfg['output_budget_bytes']
        # Supplied snapshots must explicitly contain the exact output path. Never
        # borrow source-volume free space, or apply a synthetic disk to a live host.
        disk = [d for d in snapshot.get('disks',[]) if d.get('path')==str(parent)]
        if not disk and hardware is None:
            disk = disk_snapshot((str(parent),))
        free = disk[0].get('free_bytes') if len(disk)==1 else None
        disk_budget = _budget(free)
        estimates['workspace_disk_budget'] = disk_budget
        if estimates['workspace_disk_required_bytes']>disk_budget['effective_bytes']:
            reasons.append('insufficient_or_unknown_output_volume_disk')
        torch = snapshot.get('torch',{})
        saved_version = config.get('transformers_version')
        if saved_version is not None:
            def native_version(value):
                if not isinstance(value,str):
                    raise ValueError('invalid saved/runtime Transformers version')
                match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)',value)
                if not match:
                    raise ValueError('non-release Transformers version requires runtime-specific validation')
                return tuple(int(x) for x in match.groups())
            saved = native_version(saved_version)
            installed = native_version(torch.get('packages',{}).get('transformers'))
            if saved[0]!=installed[0] or saved>installed:
                reasons.append('source_config_incompatible_with_selected_python_transformers')
        candidates = [requested]
        if requested=='auto':
            candidates = ['cuda:%d'%v['index'] for v in torch.get('devices',[])
                          if type(v.get('index')) is int and v['index']>=0]
            candidates = sorted(set(candidates),key=lambda s:int(s.split(':')[1]))+['cpu']
        rejected = {}
        estimates['cuda_candidate_budgets'] = {}
        estimates['operator_device_memory_budget_bytes'] = cfg.get('device_memory_budget_bytes')
        selected = None
        for device in candidates:
            error = _device_reason(device,torch,cfg['dtype'])
            if error is None and device.startswith('cuda:'):
                gpu = next(v for v in torch['devices'] if v['index']==int(device.split(':')[1]))
                budget = _cuda_budget(gpu, cfg.get('device_memory_budget_bytes'))
                estimates['cuda_candidate_budgets'][device] = budget
                if estimates['cuda_peak_bytes']>budget['effective_bytes']:
                    error = 'insufficient_cuda_vram'
                else:
                    estimates['cuda_budget'] = budget
            if error:
                rejected[device] = error
            else:
                selected = device
                break
        estimates['device_candidates_rejected'] = rejected
        if selected is None:
            reasons.extend('%s: %s'%(key,value) for key,value in rejected.items())
        result['execution_device'] = selected if not reasons else None
        for phase in phases:
            phase['execution_device'] = 'cpu' if phase['name']=='reconstruction' else selected
            if selected and selected.startswith('cuda:') and phase['name']!='reconstruction':
                phase['device_ram_bytes'] = estimates['cuda_phase_bytes'].get(phase['name'],0)
        synthetic = hardware is not None or snapshot.get('profile_kind')!='detected'
        result['evidence_level'] = ('synthetic_or_supplied_profile_not_hardware_validation' if synthetic else
            'cuda_implemented_requires_actual_pipeline_hardware_validation' if selected and selected.startswith('cuda:') else
            'linux_cpu_implemented_runtime_guards_required')
        unproven = []
        if not model['checkpoint_bound']:
            unproven.append('no_source_checkpoint: config-only shape estimate is not an inventory-bound execution plan')
        if result['data_profile'].get('status') != 'verified':
            unproven.append('native_train_dev_profile_unproven: no READY without complete permitted-data native encoding')
        if unproven:
            result['status'] = 'BLOCKED' if reasons else 'PLANNING_ONLY'
            reasons.extend(unproven)
            result['execution_device'] = None
        elif not reasons:
            # Bind hashes at the final planning boundary as well as around parsing.
            check_pins(result['data_profile']['_metadata']['hashes'])
            check_asset_pins(cfg['source_path'], result['data_profile']['tokenizer_file_pins'])
            result['status'] = 'READY'
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError,ImportError,RecursionError,RuntimeError) as exc:
        reasons.append('planning_refused: ' + str(exc))
        result['execution_device'] = None
    result['plan_sha256'] = digest(result)
    return result
