"""Cross-reference the human-40 eval images against any PROFESSIONAL labels we already hold:
  - TonguExpert (TE): manual expert grading in manifest.csv (*_manual columns, sparse per image).
  - TCM-Tongue (TCM): practitioner YOLO annotations (20 categories incl. red dots, fissures, zoning).
  - SM-Tongue (SM): segmentation masks only -- no characteristic labels.
"""
import json, os, pandas as pd

TCM = {0:"healthy",1:"thin_coat(non_greasy)",2:"red_tongue",3:"purple",4:"swollen",5:"thin_small",
       6:"RED_DOTS",7:"fissure",8:"toothmark",9:"white_coat",10:"yellow_coat",11:"black_coat",
       12:"slippery(greasy)",13:"kidney_cav",14:"kidney_conv",15:"liver_cav",16:"liver_conv",
       17:"spleen_cav",18:"heartlung_conv",19:"heartlung_cav"}

meta = json.load(open("data/eval/human40/meta.json"))
m = pd.read_csv("data/processed/manifest.csv")
m["base"] = m.raw_path.apply(lambda p: os.path.basename(str(p)))
counts = {"TE":0,"SM":0,"TCM":0}
te_manual = tcm_lab = 0
for e in meta:
    iid, src, base = e["id"], e["src"], os.path.basename(e["path"])
    counts[src] += 1
    if src == "TE":
        row = m[m.base == base]
        if len(row) and int(row.iloc[0].has_manual) == 1:
            r = row.iloc[0]
            labs = {k: r[k+"_manual"] for k in ["tai","zhi","fissure","tooth_mk"] if pd.notna(r[k+"_manual"])}
            if labs:
                te_manual += 1
                print(f"{iid} TE  expert-manual: {labs}")
    elif src == "TCM":
        stem = base.rsplit(".",1)[0]
        lf = f"data/external/tcm_tongue/shezhenv3-txt/train/labels/{stem}.txt"
        if os.path.exists(lf):
            cls = sorted(set(int(l.split()[0]) for l in open(lf) if l.strip()))
            tcm_lab += 1
            print(f"{iid} TCM practitioner-YOLO: {[TCM[c] for c in cls]}")

te, sm, tcm = counts["TE"], counts["SM"], counts["TCM"]
print(f"\nsource counts: {counts}")
print(f"TE with expert-manual: {te_manual}/{te}   TCM with practitioner-YOLO: {tcm_lab}/{tcm}   SM: 0/{sm} (masks only)")
