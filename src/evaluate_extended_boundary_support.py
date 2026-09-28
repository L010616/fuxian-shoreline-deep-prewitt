"""Uniform boundary-support re-evaluation of all extended-training checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

import run_extended_training_round3 as ext
import run_locked_test_model_comparison as locked
from run_multidate_ablation_pilot import predict

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'preexperiment/results/extended_training_round3'
OUT=MODEL_DIR/'unified_boundary_support'
SEEDS=ext.SEEDS

def metric(prob, truth, valid, shore):
    pred=prob>=.5
    safe=binary_erosion(valid,iterations=1)
    core=binary_erosion(valid&shore,iterations=3)
    pb_full=(pred^binary_erosion(pred))&safe
    tb_full=(truth^binary_erosion(truth))&safe
    pb=pb_full&core; tb=tb_full&core
    dt=distance_transform_edt(~tb_full)*30
    dp=distance_transform_edt(~pb_full)*30
    dpr=dt[pb]; drp=dp[tb]
    ans={'prediction_query_count':int(len(dpr)),'reference_query_count':int(len(drp))}
    for tau in (30,60):
        precision=float(np.mean(dpr<=tau)); recall=float(np.mean(drp<=tau))
        ans[f'boundary_precision_{tau}m']=precision
        ans[f'boundary_recall_{tau}m']=recall
        ans[f'boundary_f1_{tau}m']=float(2*precision*recall/(precision+recall+1e-12))
    ans['prediction_to_reference_mean_m']=float(dpr.mean())
    ans['reference_to_prediction_mean_m']=float(drp.mean())
    ans['mean_symmetric_boundary_distance_m']=float((dpr.mean()+drp.mean())/2)
    ans['hd95_m']=float(max(np.percentile(dpr,95),np.percentile(drp,95)))
    ans['prediction_to_reference_p95_m']=float(np.percentile(dpr,95))
    ans['reference_to_prediction_p95_m']=float(np.percentile(drp,95))
    return ans

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((MODEL_DIR/'protocol.json').read_text(encoding='utf8'))
    scenes=locked.load_scenes();locked.normalize_and_exclude(scenes,protocol['normalization'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allrows={}
    for name,factory in ext.factories().items():
        allrows[name]={}
        for seed in SEEDS:
            model=factory().to(device)
            cp=torch.load(MODEL_DIR/f'{name}_seed{seed}_best.pt',map_location=device,weights_only=False)
            model.load_state_dict(cp['model']); per={}
            for scene in scenes:
                per[scene['record']['l8_date']]=metric(predict(model,scene,device),scene['y'],scene['valid'],scene['shore'])
            allrows[name][str(seed)]={'checkpoint_epoch':int(cp['epoch']),'per_date':per}
            print(name,seed,flush=True)
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    keys=['boundary_f1_30m','boundary_f1_60m','mean_symmetric_boundary_distance_m','hd95_m',
          'prediction_to_reference_mean_m','reference_to_prediction_mean_m']
    summary={}
    for name in allrows:
        summary[name]={}
        for k in keys:
            vals=[]
            for seed in SEEDS:
                vals.append(float(np.mean([x[k] for x in allrows[name][str(seed)]['per_date'].values()])))
            summary[name][k]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)),'values':vals}
    payload={'status':'POST_REVIEW_UNIFIED_TARGET_SUPPORT_REEVALUATION',
      'query_support':'single-pixel boundaries within erosion(valid AND original 300 m shore, 3 pixels)',
      'target_support':'single-pixel boundaries anywhere within erosion(valid, 1 pixel)',
      'purpose':'avoid inflated nearest-neighbour distances caused by clipping target boundary to the query shore support',
      'test_status':'previously opened; no checkpoint or threshold selection used these metrics',
      'results':allrows,'summary':summary}
    (OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print('UNIFIED_SUPPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
