"""Run the single permitted evaluation on the frozen round-4 six-date blind test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
import run_multidate_ablation_pilot as ablation
import run_round4_fusion_confirmation as fusion
import run_round4_remaining_models as remaining
from evaluate_extended_boundary_support import metric as boundary_metric
from run_multidate_ablation_pilot import predict
from run_quick_model_pilot import metrics as area_metrics

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'preexperiment/results/round4_final_blind'
DRAFTS=BASE/'reference_drafts'; REF=BASE/'screened_reference'; PAIRS=BASE/'evaluation_pairs.json'
CONF=ROOT/'preexperiment/results/round4_fusion_confirmation'; REM=ROOT/'preexperiment/results/round4_remaining_models'
FREEZE=ROOT/'preexperiment/results/round4_validation_summary/round4_preblind_freeze_manifest.json'
OUT=BASE/'locked_evaluation'; SEEDS=(20260913,20260929,20261017)

def normalize_and_exclude(scenes,norm):
 for s in scenes:
  s['x']=np.empty_like(s['x_raw'],dtype='float32')
  for b,st in enumerate(norm): s['x'][b]=(np.clip(s['x_raw'][b],st['p01'],st['p99'])-st['mean'])/st['std']
  s['x'][~np.isfinite(s['x'])]=0
  s['priors']=np.stack([s['interior'],s['shore']]).astype('float32')
  with rasterio.open(REF/s['tag']/'exclusion_mask_L8_30m.tif') as src: s['valid'] &= ~src.read(1).astype(bool)

def eval_prob(prob,s):
 valid=s['valid']; shore=valid&s['shore']
 return {**area_metrics(prob,s['y'],valid,shore),**boundary_metric(prob,s['y'],valid,s['shore']),
         'valid_pixels':int(valid.sum()),'valid_shore_pixels':int(shore.sum())}

def macro(per_date,keys): return {k:float(np.mean([v[k] for v in per_date.values()])) for k in keys}

def main():
 OUT.mkdir(parents=True,exist_ok=True); result_path=OUT/'round4_final_blind_results.json'
 if result_path.exists(): raise RuntimeError('Final blind evaluation already exists; refusing to rerun')
 freeze=json.loads(FREEZE.read_text(encoding='utf-8'))
 refs=json.loads((REF/'screened_reference_manifest.json').read_text(encoding='utf-8'))
 if freeze['status']!='ALL_COMPARATOR_CONFIGS_AND_CHECKPOINTS_FROZEN_BEFORE_BLIND_TEST': raise RuntimeError('Models not frozen')
 if refs['status']!='FROZEN_FOR_ONE_TIME_FINAL_BLIND_EVALUATION': raise RuntimeError('References not frozen')
 dates=[{'l8_date':r['l8_date'],'s2_date':r['s2_date'],'split':'final_blind_test'} for r in json.loads(PAIRS.read_text(encoding='utf-8'))['selected']]
 ablation.DRAFTS=DRAFTS; scenes=[ablation.read_scene(r) for r in dates]; normalize_and_exclude(scenes,freeze_normalization())
 factories={
  'dfanet':remaining.model_factories()['dfanet'], 'prior':fusion.CONFIGS['prior'][0],
  'log':fusion.CONFIGS['log'][0], 'prewitt':remaining.model_factories()['prewitt'],
  'parallel_concat_ew0p0':fusion.CONFIGS['parallel_concat_ew0p0'][0],
  'unet_resnet18':remaining.model_factories()['unet_resnet18'],
  'deeplabv3plus_resnet18':remaining.model_factories()['deeplabv3plus_resnet18'],
  'segformer_mit_b0':remaining.model_factories()['segformer_mit_b0']}
 keys=['precision','recall','f1','iou','boundary_f1_30m','boundary_f1_60m','mean_symmetric_boundary_distance_m','hd95_m']
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); results={}
 for name,factory in factories.items():
  model_dir=CONF if name in {'prior','log','parallel_concat_ew0p0'} else REM
  results[name]={}
  for seed in SEEDS:
   model=factory().to(device); ck=torch.load(model_dir/f'{name}_seed{seed}_best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
   per={s['record']['l8_date']:eval_prob(predict(model,s,device),s) for s in scenes}
   results[name][str(seed)]={'checkpoint_epoch':int(ck['epoch']),'per_date':per,'macro':macro(per,keys)}
   print('completed',name,seed,flush=True); del model
   if torch.cuda.is_available(): torch.cuda.empty_cache()
 summary={}
 for name in results:
  summary[name]={}
  for k in keys:
   a=np.asarray([results[name][str(seed)]['macro'][k] for seed in SEEDS])
   summary[name][k]={'mean':float(a.mean()),'std':float(a.std(ddof=1)),'values':a.tolist()}
 payload={'status':'FINAL_BLIND_EVALUATION_OPENED_ONCE_AND_LOCKED','models':list(results),'seeds':list(SEEDS),
  'dates':[r['l8_date'] for r in dates],'decision_threshold':0.5,'selected_primary_model':'prior',
  'training_or_model_selection_after_test':False,'reference_status':'Conservatively screened Sentinel-2 shoreline reference; not surveyed ground truth',
  'all_six_prefrozen_dates_included':True,'device':str(device),'results':results,'summary':summary}
 result_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({n:{'BF1_30m':round(s['boundary_f1_30m']['mean'],4),'ASD_m':round(s['mean_symmetric_boundary_distance_m']['mean'],2),'IoU':round(s['iou']['mean'],4)} for n,s in summary.items()},indent=2),flush=True)

def freeze_normalization():
 return json.loads((CONF/'fusion_confirmation_results.json').read_text(encoding='utf-8'))['normalization']
if __name__=='__main__': main()
