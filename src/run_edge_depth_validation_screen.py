"""Validation-only screen for deeper edge encoders.

This script deliberately never loads the opened final-blind-test directory.
Architecture selection is restricted to the existing training/validation dates.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_multidate_ablation_pilot as ablation
import run_round4_fusion_screen as screen
from run_multidate_ablation_pilot import FixedEdgeMode, MultiSceneDataset, seed_all, windows_for
from run_quick_model_pilot import Encoder, ConvBNAct, DSConv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preexperiment/results/edge_depth_validation_screen'
SEEDS=(20260913,20260929,20261017)
MAX_EPOCHS=40; MIN_EPOCHS=20; PATIENCE=8

class ResidualRefine(nn.Module):
    def __init__(self, channels:int, depth:int):
        super().__init__(); self.blocks=nn.ModuleList([DSConv(channels,channels) for _ in range(depth)])
    def forward(self,x):
        for block in self.blocks: x=x+block(x)
        return x

class EdgeEncoder(nn.Module):
    def __init__(self, depth:int):
        super().__init__(); widths=(12,20,32,48)
        self.stem=ConvBNAct(4,widths[0],3,2); self.r0=ResidualRefine(widths[0],depth)
        self.s1=DSConv(widths[0],widths[1],2); self.r1=ResidualRefine(widths[1],depth)
        self.s2=DSConv(widths[1],widths[2],2); self.r2=ResidualRefine(widths[2],depth)
        self.s3=DSConv(widths[2],widths[3],2); self.r3=ResidualRefine(widths[3],depth)
    def forward(self,x):
        a=self.r0(self.stem(x)); b=self.r1(self.s1(a)); c=self.r2(self.s2(b)); d=self.r3(self.s3(c)); return a,b,c,d

class DeeperEdgeDFANet(nn.Module):
    def __init__(self, depth:int):
        super().__init__(); self.prior=True; self.edge_mode='parallel'; self.depth=depth
        self.semantic=Encoder(8,(24,48,96,128)); self.edges=FixedEdgeMode('parallel')
        self.edge_encoder=EdgeEncoder(depth); ew=(12,20,32,48)
        self.d2=DSConv(128+ew[3]+96+ew[2],96); self.d1=DSConv(96+48+ew[1],64); self.d0=DSConv(64+24+ew[0],48)
        self.head=nn.Conv2d(48,1,1)
    def forward(self,x,priors):
        s0,s1,s2,s3=self.semantic(torch.cat([x,priors],1)); e0,e1,e2,e3=self.edge_encoder(self.edges(x))
        z=torch.cat([s3,e3],1)
        z=F.interpolate(z,size=s2.shape[-2:],mode='bilinear',align_corners=False); z=self.d2(torch.cat([z,s2,e2],1))
        z=F.interpolate(z,size=s1.shape[-2:],mode='bilinear',align_corners=False); z=self.d1(torch.cat([z,s1,e1],1))
        z=F.interpolate(z,size=s0.shape[-2:],mode='bilinear',align_corners=False); z=self.d0(torch.cat([z,s0,e0],1))
        return F.interpolate(self.head(z),size=x.shape[-2:],mode='bilinear',align_corners=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result_path=OUT/'edge_depth_validation_results.json'
    if result_path.exists(): raise RuntimeError('Validation depth screen already exists; refusing to overwrite')
    scenes=screen.load_scenes(); train=[s for s in scenes if s['record']['split']=='train']; val=[s for s in scenes if s['record']['split']=='validation']
    norm=ablation.normalize_scenes(scenes); tw=windows_for(train); vw=windows_for(val)
    vl=DataLoader(MultiSceneDataset(val,vw),batch_size=12,shuffle=False,num_workers=0,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); results={}
    for depth in (0,1,2):
      name=f'edge_depth_{depth}'; results[name]={}
      for seed in SEEDS:
        seed_all(seed); tl=DataLoader(MultiSceneDataset(train,tw,augment=True),batch_size=12,shuffle=True,num_workers=0,pin_memory=True)
        model=DeeperEdgeDFANet(depth).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf'); stale=0; hist=[]; ckpt=OUT/f'{name}_seed{seed}_best.pt'
        for epoch in range(1,MAX_EPOCHS+1):
          tr=screen.train_epoch(model,tl,device,opt,0.0); va=screen.validation_loss(model,vl,device,0.0); hist.append({'epoch':epoch,'train_loss':tr,'validation_loss':va})
          if va<best-1e-4: best=va; stale=0; torch.save({'model':model.state_dict(),'epoch':epoch,'depth':depth},ckpt)
          else: stale+=1
          if epoch>=MIN_EPOCHS and stale>=PATIENCE: break
        state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state['model'])
        per,macro=screen.validate_full_scenes(model,val,device)
        results[name][str(seed)]={'best_epoch':state['epoch'],'best_validation_loss':best,'parameters':sum(p.numel() for p in model.parameters()),'per_date':per,'validation_macro':macro,'history':hist}
        print(name,seed,'epoch',state['epoch'],'BF1',round(macro['boundary_f1_30m'],5),'ASD',round(macro['mean_symmetric_boundary_distance_m'],2),flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    keys=('iou','boundary_f1_30m','boundary_f1_60m','mean_symmetric_boundary_distance_m','hd95_m'); summary={}
    for name,rr in results.items():
      summary[name]={'parameters':rr[str(SEEDS[0])]['parameters']}
      for k in keys:
        a=np.array([rr[str(s)]['validation_macro'][k] for s in SEEDS]); summary[name][k]={'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'values':a.tolist()}
    ranking=sorted(summary,key=lambda n:(-summary[n]['boundary_f1_30m']['mean'],summary[n]['mean_symmetric_boundary_distance_m']['mean']))
    payload={'status':'VALIDATION_ONLY_EDGE_DEPTH_SCREEN','final_blind_test_accessed':False,'architecture_selection_allowed_only_on_validation':True,'seeds':SEEDS,'training_dates':[s['record']['l8_date'] for s in train],'validation_dates':[s['record']['l8_date'] for s in val],'training_windows':len(tw),'validation_windows':len(vw),'normalization':norm,'ranking_rule':'highest validation BF1 at 30 m; tie-break lower ASD','ranking':ranking,'summary':summary,'results':results}
    result_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'edge_depth_validation_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
      w=csv.writer(f); w.writerow(['rank','configuration','extra_blocks_per_scale','parameters','BF1_30m_mean','BF1_30m_sd','ASD_m_mean','ASD_m_sd','IoU_mean'])
      for rank,name in enumerate(ranking,1):
        s=summary[name]; w.writerow([rank,name,int(name.rsplit('_',1)[1]),s['parameters'],s['boundary_f1_30m']['mean'],s['boundary_f1_30m']['sd'],s['mean_symmetric_boundary_distance_m']['mean'],s['mean_symmetric_boundary_distance_m']['sd'],s['iou']['mean']])
    print(json.dumps({'ranking':ranking,'summary':summary},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
