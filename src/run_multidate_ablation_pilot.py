"""Multi-date five-variant ablation pilot with date-level validation.

This is a fast feasibility experiment. Labels are AI-screened S2 drafts and the
network is a compact DFANet-style implementation, so results are not final SCI
accuracy claims.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import label as component_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from run_quick_model_pilot import Encoder, DSConv, masked_loss, metrics


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preexperiment"
DATA_ROOT = ROOT
DRAFTS = PRE / "results/formal_trainval_drafts_v02"
OUT = PRE / "results/multidate_ablation_pilot"
MANIFEST = PRE / "results/pilot_manifest_paired.json"
SEED = 20260913
PATCH = 96
STRIDE = 64
EPOCHS = 15
L8_TEMPLATE = "Fuxian_L8_native_{tag}.tif"
MASK_TEMPLATE = "Fuxian_masks_native_{tag}.tif"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def compact(s):
    return s.replace("-", "")


def fixed_interior(roi, shore):
    labels, count = component_label(roi & ~shore)
    if count == 0:
        raise RuntimeError("Spatial prior has no interior component")
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == int(np.argmax(sizes))


def read_scene(record):
    tag = compact(record["l8_date"])
    l8p = DATA_ROOT / L8_TEMPLATE.format(tag=tag)
    maskp = DATA_ROOT / MASK_TEMPLATE.format(tag=tag)
    draftp = DRAFTS / tag / "S2_water_DRAFT.tif"
    with rasterio.open(l8p) as src:
        x = src.read(range(1, 7)).astype("float32"); profile = src.profile.copy()
        l8_transform = src.transform
        crs = src.crs
    with rasterio.open(maskp) as src:
        masks = src.read()
    with rasterio.open(draftp) as src:
        water = src.read(1)
        fraction = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
        source = np.where(water == 255, np.nan, water).astype("float32")
        reproject(source, fraction,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=l8_transform, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.average)
    y = fraction >= 0.5
    finite = np.all(np.isfinite(x) & (x > -9000), axis=0)
    shore, joint, roi = masks[0].astype(bool), masks[3].astype(bool), masks[4].astype(bool)
    valid = finite & joint & np.isfinite(fraction) & roi
    interior = fixed_interior(roi, shore)
    return {"tag": tag, "record": record, "x_raw": x, "y": y, "valid": valid,
            "shore": shore, "roi": roi, "interior": interior, "profile": profile}


def normalize_scenes(scenes):
    stats = []
    for b in range(6):
        vals = np.concatenate([s["x_raw"][b][s["valid"]] for s in scenes if s["record"]["split"] == "train"])
        lo, hi = np.percentile(vals, [1, 99])
        clipped = np.clip(vals, lo, hi); mean, std = float(clipped.mean()), float(clipped.std())
        std = max(std, 1e-6)
        stats.append({"band": b + 1, "p01": float(lo), "p99": float(hi), "mean": mean, "std": std})
        for s in scenes:
            s.setdefault("x", np.empty_like(s["x_raw"], dtype="float32"))
            s["x"][b] = (np.clip(s["x_raw"][b], lo, hi) - mean) / std
    for s in scenes:
        s["x"][~np.isfinite(s["x"])] = 0
        s["priors"] = np.stack([s["interior"], s["shore"]]).astype("float32")
    return stats


def windows_for(scenes):
    rows = []
    for si, s in enumerate(scenes):
        h, w = s["y"].shape
        for r in range(0, h - PATCH + 1, STRIDE):
            for c in range(0, w - PATCH + 1, STRIDE):
                sl = np.s_[r:r + PATCH, c:c + PATCH]
                if (s["valid"][sl] & s["shore"][sl]).sum() >= 80:
                    rows.append((si, r, c))
    return rows


class MultiSceneDataset(Dataset):
    def __init__(self, scenes, windows, augment=False):
        self.scenes, self.windows, self.augment = scenes, windows, augment
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        si, r, c = self.windows[idx]; s = self.scenes[si]; sl = np.s_[r:r+PATCH, c:c+PATCH]
        x = s["x"][:, sl[0], sl[1]].copy(); priors = s["priors"][:, sl[0], sl[1]].copy()
        y = s["y"][sl].astype("float32")[None]
        mask = s["valid"][sl].astype("float32")[None]
        # Give the uncertain shoreline band more influence without changing labels.
        mask *= np.where(s["shore"][sl], 4.0, 1.0)[None]
        if self.augment:
            k = random.randrange(4)
            x, priors, y, mask = [np.rot90(a, k, axes=(-2, -1)).copy() for a in (x, priors, y, mask)]
            if random.random() < 0.5:
                x, priors, y, mask = [a[..., ::-1].copy() for a in (x, priors, y, mask)]
        return tuple(torch.from_numpy(a) for a in (x, priors, y, mask))


class FixedEdgeMode(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        px = torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=torch.float32) / 3
        log = torch.tensor([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=torch.float32) / 16
        self.register_buffer("px", px[None,None]); self.register_buffer("py", px.t()[None,None]); self.register_buffer("log", log[None,None])
    def forward(self, x):
        out = []
        for channel in (x[:,1:2], x[:,3:4]):
            if self.mode in ("prewitt", "parallel"):
                gx=F.conv2d(channel,self.px,padding=1); gy=F.conv2d(channel,self.py,padding=1)
                out.append(torch.sqrt(gx.square()+gy.square()+1e-6))
            if self.mode in ("log", "parallel"):
                out.append(torch.abs(F.conv2d(channel,self.log,padding=2)))
        return torch.cat(out,1)


class AblationDFANet(nn.Module):
    def __init__(self, prior=False, edge_mode=None):
        super().__init__(); self.prior=prior; self.edge_mode=edge_mode
        self.semantic=Encoder(8 if prior else 6,(24,48,96,128))
        if edge_mode:
            edge_in=4 if edge_mode=="parallel" else 2
            self.edges=FixedEdgeMode(edge_mode); self.edge_encoder=Encoder(edge_in,(12,20,32,48)); ew=(12,20,32,48)
        else: ew=(0,0,0,0)
        self.d2=DSConv(128+ew[3]+96+ew[2],96); self.d1=DSConv(96+48+ew[1],64); self.d0=DSConv(64+24+ew[0],48)
        self.head=nn.Conv2d(48,1,1)
    def forward(self,x,priors):
        s=self.semantic(torch.cat([x,priors],1) if self.prior else x); s0,s1,s2,s3=s
        if self.edge_mode:
            e0,e1,e2,e3=self.edge_encoder(self.edges(x)); z=torch.cat([s3,e3],1)
            z=F.interpolate(z,size=s2.shape[-2:],mode="bilinear",align_corners=False); z=self.d2(torch.cat([z,s2,e2],1))
            z=F.interpolate(z,size=s1.shape[-2:],mode="bilinear",align_corners=False); z=self.d1(torch.cat([z,s1,e1],1))
            z=F.interpolate(z,size=s0.shape[-2:],mode="bilinear",align_corners=False); z=self.d0(torch.cat([z,s0,e0],1))
        else:
            z=F.interpolate(s3,size=s2.shape[-2:],mode="bilinear",align_corners=False); z=self.d2(torch.cat([z,s2],1))
            z=F.interpolate(z,size=s1.shape[-2:],mode="bilinear",align_corners=False); z=self.d1(torch.cat([z,s1],1))
            z=F.interpolate(z,size=s0.shape[-2:],mode="bilinear",align_corners=False); z=self.d0(torch.cat([z,s0],1))
        return F.interpolate(self.head(z),size=x.shape[-2:],mode="bilinear",align_corners=False)


def epoch(model, loader, device, optimizer=None):
    train=optimizer is not None; model.train(train); values=[]
    for x,p,y,m in loader:
        x,p,y,m=[a.to(device,non_blocking=True) for a in (x,p,y,m)]
        with torch.set_grad_enabled(train):
            loss=masked_loss(model(x,p),y,m)
            if train: optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        values.append(float(loss.detach()))
    return float(np.mean(values))


@torch.no_grad()
def predict(model,s,device,tile=256,overlap=32):
    model.eval(); h,w=s["y"].shape; total=np.zeros((h,w),"float32"); count=np.zeros((h,w),"float32"); step=tile-overlap
    rows=list(range(0,max(1,h-tile+1),step)); cols=list(range(0,max(1,w-tile+1),step))
    if rows[-1]!=h-tile: rows.append(max(0,h-tile))
    if cols[-1]!=w-tile: cols.append(max(0,w-tile))
    for r in rows:
        for c in cols:
            x=torch.from_numpy(s["x"][:,r:r+tile,c:c+tile][None]).to(device); p=torch.from_numpy(s["priors"][:,r:r+tile,c:c+tile][None]).to(device)
            q=torch.sigmoid(model(x,p))[0,0].cpu().numpy(); total[r:r+tile,c:c+tile]+=q; count[r:r+tile,c:c+tile]+=1
    return total/np.maximum(count,1)


def main():
    seed_all(SEED); OUT.mkdir(parents=True,exist_ok=True)
    records=json.loads(MANIFEST.read_text(encoding="utf-8"))["records"]
    selected=[r for r in records if r["split"] in ("train","validation")]
    scenes=[read_scene(r) for r in selected]; train=[s for s in scenes if s["record"]["split"]=="train"]; val=[s for s in scenes if s["record"]["split"]=="validation"]
    stats=normalize_scenes(scenes); tw=windows_for(train); vw=windows_for(val)
    loaders={"train":DataLoader(MultiSceneDataset(train,tw,True),batch_size=24,shuffle=True,num_workers=0,pin_memory=True),
             "val":DataLoader(MultiSceneDataset(val,vw,False),batch_size=24,shuffle=False,num_workers=0,pin_memory=True)}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants={"dfanet":(False,None),"prior":(True,None),"log":(True,"log"),"prewitt":(True,"prewitt"),"parallel":(True,"parallel")}
    logs=[]; results={}
    for name,(prior,edge_mode) in variants.items():
        seed_all(SEED); model=AblationDFANet(prior,edge_mode).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=2,factor=.5); best=math.inf; patience=0; started=time.time()
        for ep in range(1,EPOCHS+1):
            tr=epoch(model,loaders["train"],device,opt); va=epoch(model,loaders["val"],device); scheduler.step(va)
            logs.append({"model":name,"epoch":ep,"train_loss":tr,"val_loss":va}); print(f"{name} epoch={ep:02d} train={tr:.4f} val={va:.4f}",flush=True)
            if va<best-1e-4: best=va;patience=0;torch.save({"model":model.state_dict(),"epoch":ep,"val_loss":va},OUT/f"{name}_best.pt")
            else: patience+=1
            if ep>=8 and patience>=4: break
        ck=torch.load(OUT/f"{name}_best.pt",map_location=device,weights_only=False);model.load_state_dict(ck["model"])
        per_date={}
        for s in val:
            prob=predict(model,s,device)
            if prior:
                prob[s["interior"]]=1; prob[s["roi"]&~s["shore"]&~s["interior"]]=0; prob[~s["roi"]]=0
            per_date[s["record"]["l8_date"]]=metrics(prob,s["y"],s["valid"],s["valid"]&s["shore"])
        keys=("precision","recall","f1","iou","boundary_f1_30m","boundary_f1_60m","mean_symmetric_boundary_distance_m")
        macro={k:float(np.mean([v[k] for v in per_date.values()])) for k in keys}
        results[name]={"best_epoch":int(ck["epoch"]),"best_val_loss":float(ck["val_loss"]),"parameters":sum(p.numel() for p in model.parameters()),"seconds":time.time()-started,"macro":macro,"per_date":per_date}
    with (OUT/"training_log.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["model","epoch","train_loss","val_loss"]);w.writeheader();w.writerows(logs)
    config={"purpose":"multi-date ablation feasibility pilot","not_final_paper_results":True,"seed":SEED,"train_dates":[s["record"]["l8_date"] for s in train],"validation_dates":[s["record"]["l8_date"] for s in val],"patch":PATCH,"stride":STRIDE,"train_patches":len(tw),"validation_patches":len(vw),"epochs_max":EPOCHS,"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"labels":"AI-screened automatic S2 NDWI outer-boundary drafts","normalization":stats}
    (OUT/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"metrics.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 多日期五组消融预实验","","本结果使用 6 个训练日期和 2 个独立验证日期。标签仍为 S2 自动草稿，网络为紧凑 DFANet 风格实现，因此不能作为论文最终精度。","","|模型|IoU|F1|Boundary F1 30 m|Boundary F1 60 m|平均岸线距离/m|","|---|---:|---:|---:|---:|---:|"]
    labels={"dfanet":"DFANet","prior":"边界约束","log":"边界约束+LoG","prewitt":"边界约束+Prewitt","parallel":"完整并联"}
    for n in variants:
        m=results[n]["macro"];lines.append(f"|{labels[n]}|{m['iou']:.4f}|{m['f1']:.4f}|{m['boundary_f1_30m']:.4f}|{m['boundary_f1_60m']:.4f}|{m['mean_symmetric_boundary_distance_m']:.2f}|")
    lines += ["","验证日期完全独立于训练日期。正式实验仍需人工定稿参考岸线、标准网络对比及 3 个随机种子。"]
    (OUT/"多日期消融预实验报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"config":config,"macro":{k:v["macro"] for k,v in results.items()}},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
