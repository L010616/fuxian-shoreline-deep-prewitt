"""One-time locked evaluation on the six Prewitt confirmation set 3 confirmatory dates."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import rasterio
import torch
from scipy.stats import wilcoxon

import run_multidate_ablation_pilot as ablation
import run_round4_fusion_confirmation as fusion
from evaluate_extended_boundary_support import metric as boundary_metric
from run_edge_depth_validation_screen import DeeperEdgeDFANet
from run_deep_edge_component_ablation import DeepSingleOperatorDFANet
from run_parameter_matched_conv_control import ParameterMatchedConvDFANet
from run_multidate_ablation_pilot import predict
from run_quick_model_pilot import metrics as area_metrics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'preexperiment/results/prewitt_confirmation_set3'
DATA=ROOT/'preexperiment/data/prewitt_confirmation_set3/analysis_ready'
DRAFTS=BASE/'reference_drafts'; REF=BASE/'screened_reference'; PAIRS=BASE/'evaluation_pairs.json'
FREEZE=BASE/'prewitt_confirmation_set3_evaluation_freeze_manifest.json'
DEPTH=ROOT/'preexperiment/results/edge_depth_validation_screen'
COMPONENT=ROOT/'preexperiment/results/deep_edge_component_ablation'
PRIOR=ROOT/'preexperiment/results/round4_fusion_confirmation'
CONTROL=ROOT/'preexperiment/results/parameter_matched_conv_control'
OUT=BASE/'locked_evaluation'; SEEDS=(20260913,20260929,20261017)
KEYS=('precision','recall','f1','iou','boundary_f1_30m','boundary_f1_60m','mean_symmetric_boundary_distance_m','hd95_m')

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def normalize(scenes,norm):
    for s in scenes:
      s['x']=np.empty_like(s['x_raw'],dtype='float32')
      for b,z in enumerate(norm): s['x'][b]=(np.clip(s['x_raw'][b],z['p01'],z['p99'])-z['mean'])/z['std']
      s['x'][~np.isfinite(s['x'])]=0; s['priors']=np.stack([s['interior'],s['shore']]).astype('float32')
      with rasterio.open(REF/s['tag']/'exclusion_mask_L8_30m.tif') as src: s['valid'] &= ~src.read(1).astype(bool)

def eval_prob(p,s):
    v=s['valid']; sh=v&s['shore']
    return {**area_metrics(p,s['y'],v,sh),**boundary_metric(p,s['y'],v,s['shore']),
            'valid_pixels':int(v.sum()),'valid_shore_pixels':int(sh.sum())}

def macro(per): return {k:float(np.mean([r[k] for r in per.values()])) for k in KEYS}

def effect(candidate,comparator,key,positive_if_lower=False,iters=20000):
    dates=sorted(candidate)
    vals=np.asarray([(comparator[d][key]-candidate[d][key]) if positive_if_lower else
                     (candidate[d][key]-comparator[d][key]) for d in dates],dtype='float64')
    rng=np.random.default_rng(20260922); boot=vals[rng.integers(0,len(vals),size=(iters,len(vals)))].mean(1)
    try: p=float(wilcoxon(vals,alternative='two-sided',method='exact').pvalue)
    except ValueError: p=1.0
    return {'date_level_improvement':{d:float(v) for d,v in zip(dates,vals)},'mean_improvement':float(vals.mean()),
      'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
      'dates_improved':int((vals>0).sum()),'dates_tied':int((vals==0).sum()),'dates_worsened':int((vals<0).sum()),
      'wilcoxon_exact_two_sided_p':p}

def main():
    OUT.mkdir(parents=True,exist_ok=True); out=OUT/'prewitt_confirmation_set3_results.json'
    if out.exists(): raise RuntimeError('Locked Prewitt confirmation set 3 result exists; refusing to rerun')
    freeze=json.loads(FREEZE.read_text(encoding='utf-8')); refs=json.loads((REF/'screened_reference_manifest.json').read_text(encoding='utf-8')); pairs=json.loads(PAIRS.read_text(encoding='utf-8'))
    if freeze['status']!='MODELS_REFERENCES_AND_DATES_FROZEN_BEFORE_PREWITT_CONFIRMATION_SET3_PREDICTIONS': raise RuntimeError('Not frozen')
    if refs['status']!='FROZEN_FOR_ONE_TIME_PREWITT_CONFIRMATION_SET3_EVALUATION': raise RuntimeError('References not frozen')
    if sha256(REF/'screened_reference_manifest.json')!=freeze['reference_manifest_sha256']:
      raise RuntimeError('Reference manifest changed after freeze')
    if sha256(PAIRS)!=freeze['pairs_manifest_sha256']:
      raise RuntimeError('Date-pair manifest changed after freeze')
    for x in freeze['checkpoints']:
      if sha256(ROOT/x['path'])!=x['sha256']: raise RuntimeError('Checkpoint hash mismatch: '+x['path'])
    depth=json.loads((DEPTH/'edge_depth_validation_results.json').read_text(encoding='utf-8'))
    component=json.loads((COMPONENT/'deep_edge_component_ablation_results.json').read_text(encoding='utf-8'))
    control=json.loads((CONTROL/'parameter_matched_conv_validation.json').read_text(encoding='utf-8'))
    norm=depth['normalization']
    a=np.array([[z[k] for k in ('p01','p99','mean','std')] for z in norm]); b=np.array([[z[k] for k in ('p01','p99','mean','std')] for z in control['normalization']])
    if not np.allclose(a,b,rtol=0,atol=1e-12): raise RuntimeError('Normalization mismatch')
    c=np.array([[z[k] for k in ('p01','p99','mean','std')] for z in component['normalization']])
    if not np.allclose(a,c,rtol=0,atol=1e-12): raise RuntimeError('Prewitt normalization mismatch')
    ablation.DATA_ROOT=DATA; ablation.DRAFTS=DRAFTS
    ablation.L8_TEMPLATE='Fuxian_prewitt_confirm_set3_L8_native_{tag}.tif'
    ablation.MASK_TEMPLATE='Fuxian_prewitt_confirm_set3_masks_native_{tag}.tif'
    dates=[{'l8_date':r['l8_date'],'s2_date':r['s2_date'],'split':'prewitt_confirmation_set3_test'} for r in pairs['selected']]
    scenes=[ablation.read_scene(r) for r in dates]; normalize(scenes,norm)
    factories={'prior_only':fusion.CONFIGS['prior'][0],
      'fixed_prewitt_deep':lambda:DeepSingleOperatorDFANet('prewitt'),
      'fixed_log_prewitt_deep':lambda:DeeperEdgeDFANet(2),
      'parameter_matched_ordinary_conv':ParameterMatchedConvDFANet}
    checkpoints={'prior_only':lambda s:PRIOR/f'prior_seed{s}_best.pt',
      'fixed_prewitt_deep':lambda s:COMPONENT/f'deep_prewitt_only_seed{s}_best.pt',
      'fixed_log_prewitt_deep':lambda s:DEPTH/f'edge_depth_2_seed{s}_best.pt',
      'parameter_matched_ordinary_conv':lambda s:CONTROL/f'matched_conv_seed{s}_best.pt'}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); results={}
    for name,factory in factories.items():
      results[name]={}
      for seed in SEEDS:
        model=factory().to(device); state=torch.load(checkpoints[name](seed),map_location=device,weights_only=False); model.load_state_dict(state['model'])
        per={s['record']['l8_date']:eval_prob(predict(model,s,device),s) for s in scenes}
        results[name][str(seed)]={'checkpoint_epoch':int(state['epoch']),'per_date':per,'macro':macro(per)}
        print('completed',name,seed,flush=True); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary={}; avg={}
    for name,by in results.items():
      summary[name]={k:{'mean':float(np.mean([by[str(s)]['macro'][k] for s in SEEDS])),
                         'sd':float(np.std([by[str(s)]['macro'][k] for s in SEEDS],ddof=1))} for k in KEYS}
      avg[name]={d:{k:float(np.mean([by[str(s)]['per_date'][d][k] for s in SEEDS])) for k in KEYS} for d in [r['l8_date'] for r in pairs['selected']]}
    comparisons={}
    for comp in ('prior_only','fixed_log_prewitt_deep','parameter_matched_ordinary_conv'):
      comparisons['fixed_prewitt_deep_vs_'+comp]={
       'boundary_f1_30m':effect(avg['fixed_prewitt_deep'],avg[comp],'boundary_f1_30m'),
       'asd_reduction_m':effect(avg['fixed_prewitt_deep'],avg[comp],'mean_symmetric_boundary_distance_m',True)}
    payload={'status':'PREWITT_CONFIRMATION_SET3_EVALUATION_OPENED_ONCE_AND_LOCKED','models':list(factories),'seeds':list(SEEDS),
      'dates':[r['l8_date'] for r in pairs['selected']],'decision_threshold':.5,'all_six_frozen_dates_included':True,
      'reference_status':'Prediction-blind screened Sentinel-2 shoreline reference; not surveyed ground truth',
      'rejected_before_predictions':pairs.get('rejected_before_predictions',[]),
      'training_model_selection_or_threshold_tuning_after_test':False,'device':str(device),'results':results,
      'summary':summary,'seed_averaged_per_date':avg,'paired_date_level_effects':comparisons}
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'per_date_seed_averaged.csv').open('w',encoding='utf-8-sig',newline='') as f:
      w=csv.writer(f);w.writerow(['model','date',*KEYS])
      for n,dd in avg.items():
       for d,row in dd.items():w.writerow([n,d,*[row[k] for k in KEYS]])
    print(json.dumps({'summary':{n:{'BF1':round(v['boundary_f1_30m']['mean'],6),'ASD_m':round(v['mean_symmetric_boundary_distance_m']['mean'],3),'IoU':round(v['iou']['mean'],6)} for n,v in summary.items()},'comparisons':comparisons},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()


