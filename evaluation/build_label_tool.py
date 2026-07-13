"""Build a SELF-CONTAINED labeling page: embeds the human-40 images as base64 data URIs so the tool
works by just double-clicking the HTML file — no web server, no external image files.

    python3 evaluation/build_label_tool.py
    -> writes evaluation/label_human40.html (open it directly in a browser)

Regenerate whenever the eval image set changes. Images are downscaled (max 1000px, JPEG q82) to keep
the file small while staying sharp enough to label.
"""
import argparse, base64, glob, io, json, os, re
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MAXDIM, QUALITY = 1000, 82

# --- field schemas ---
FIELDS_EXTRA = [
    {"key": "red_tip", "lab": "Red tip", "help": "Is the tip visibly redder than the rest of the body?", "opts": ["none", "mild", "strong"]},
    {"key": "red_dots", "lab": "Red dots / prickles", "help": "Red dots or raised red papillae on the surface.", "opts": ["none", "few", "many"]},
    {"key": "surface_pattern", "lab": "Surface pattern / texture", "help": "Texture so patterned you can't tell if it's greasy.", "opts": ["none", "present"]},
    {"key": "coating_obscures_body", "lab": "Coating hides body colour", "help": "Centre coating too thick/pale to read the body colour.", "opts": ["no", "yes"]},
    {"key": "tip_shape_ambiguous", "lab": "Tip shape ambiguous", "help": "Odd tip shape — could be a tooth-mark or just anatomy.", "opts": ["no", "yes"]},
]
FIELDS_CORE = [
    {"key": "coating", "lab": "Coating (greasiness)", "help": "How greasy/thick is the coating?", "opts": ["non_greasy", "greasy", "greasy_thick"]},
    {"key": "tai", "lab": "Coating colour", "help": "Colour of the coating film.", "opts": ["white", "light_yellow", "yellow"]},
    {"key": "zhi", "lab": "Body colour", "help": "Colour of the tongue body itself (ignore coating). Leave blank if unreadable.", "opts": ["light", "regular", "dark"]},
    {"key": "fissure", "lab": "Fissures / cracks", "help": "Cracks in the surface.", "opts": ["none", "light", "severe"]},
    {"key": "tooth_mk", "lab": "Tooth marks", "help": "Scalloped indentations on the edges.", "opts": ["none", "light", "severe"]},
]
SCHEMAS = {"extra": FIELDS_EXTRA, "full": FIELDS_CORE + FIELDS_EXTRA}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/eval/human40", help="image dir (repo-relative)")
ap.add_argument("--out", default=None, help="output html (default: evaluation/label_<set>.html)")
ap.add_argument("--mode", choices=list(SCHEMAS), default="extra")
ap.add_argument("--exclude", default="", help="comma ids to drop, e.g. t15,t22")
ap.add_argument("--flags", default="", help="id:reason;... e.g. t24:rotated")
ap.add_argument("--ls-key", default=None, help="localStorage key (default from set name)")
args = ap.parse_args()

IMG_DIR = os.path.join(REPO, args.dir)
SET = os.path.basename(args.dir.rstrip("/"))
OUT = os.path.join(REPO, args.out) if args.out else os.path.join(HERE, f"label_{SET}.html")
FIELDS = SCHEMAS[args.mode]
LS_KEY = args.ls_key or f"{SET}_{args.mode}_labels_v1"
EXCLUDE = set(x for x in args.exclude.split(",") if x)
FLAGS = dict(kv.split(":") for kv in args.flags.split(";") if ":" in kv)
IDS = sorted(re.sub(r"\.(jpg|jpeg|png)$", "", os.path.basename(p))
             for p in glob.glob(os.path.join(IMG_DIR, "t*.*"))
             if p.lower().endswith((".jpg", ".jpeg", ".png")))
IDS = [i for i in IDS if i not in EXCLUDE]


def encode(path):
    im = Image.open(path).convert("RGB")
    s = MAXDIM / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


images = {}
for iid in IDS:
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(IMG_DIR, iid + ext)
        if os.path.exists(p):
            images[iid] = encode(p)
            break
    else:
        print(f"WARN: no image for {iid}")

# JS-embeddable map. Data URIs are ASCII-safe, so a plain join is fine.
img_js = "{\n" + ",\n".join(f'"{k}":"{v}"' for k, v in images.items()) + "\n}"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tongue labeling — __SET__</title>
<style>
  :root { --bg:#f6f5f2; --card:#fff; --ink:#1c1a17; --mut:#7a736a; --line:#e2ddd4;
          --accent:#c0392b; --accent2:#2d7d5a; --sel:#1c1a17; }
  @media (prefers-color-scheme: dark){
    :root { --bg:#17150f; --card:#211e18; --ink:#efe9df; --mut:#9a9184; --line:#332e25;
            --accent:#e05a48; --accent2:#4bbd8a; --sel:#efe9df; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
           padding:10px 18px; display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:650; }
  .prog { flex:1; min-width:160px; height:8px; background:var(--line); border-radius:5px; overflow:hidden; }
  .prog > div { height:100%; background:var(--accent2); width:0%; transition:width .2s; }
  .progtxt { font-variant-numeric:tabular-nums; color:var(--mut); font-size:13px; }
  main { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr); gap:22px;
         max-width:1100px; margin:18px auto; padding:0 18px; align-items:start; }
  .imgwrap { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px;
             position:sticky; top:64px; }
  .imgwrap img { width:100%; border-radius:8px; display:block; background:#000; }
  .idbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
  .idbar .iid { font-weight:700; font-size:18px; letter-spacing:.5px; }
  .idbar .flag { font-size:12px; color:var(--accent); }
  .fields { display:flex; flex-direction:column; gap:16px; }
  .field { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:13px 14px; }
  .field .lab { font-weight:640; }
  .field .help { color:var(--mut); font-size:12.5px; margin:2px 0 9px; }
  .opts { display:flex; gap:8px; flex-wrap:wrap; }
  .opts button { flex:1 1 auto; min-width:64px; padding:8px 10px; border:1.5px solid var(--line);
                 background:transparent; color:var(--ink); border-radius:9px; cursor:pointer; font-size:14px; }
  .opts button:hover { border-color:var(--mut); }
  .opts button.on { background:var(--sel); color:var(--bg); border-color:var(--sel); font-weight:650; }
  nav { display:flex; gap:10px; align-items:center; margin-top:4px; }
  nav button { padding:9px 16px; border:1.5px solid var(--line); background:var(--card); color:var(--ink);
               border-radius:9px; cursor:pointer; font-size:14px; }
  nav button:hover { border-color:var(--mut); }
  nav .spacer { flex:1; }
  .foot { max-width:1100px; margin:8px auto 40px; padding:0 18px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .foot button { padding:10px 16px; border-radius:9px; cursor:pointer; border:1.5px solid var(--line);
                 background:var(--card); color:var(--ink); font-size:14px; }
  .foot .primary { background:var(--accent2); border-color:var(--accent2); color:#fff; font-weight:650; }
  .hint { color:var(--mut); font-size:12.5px; }
  .grid-jump { display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }
  .grid-jump button { width:34px; height:28px; padding:0; font-size:11px; border:1px solid var(--line);
                      background:var(--card); color:var(--mut); border-radius:6px; cursor:pointer; }
  .grid-jump button.done { background:var(--accent2); color:#fff; border-color:var(--accent2); }
  .grid-jump button.cur { outline:2px solid var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Tongue labeling · <span style="color:var(--mut);font-weight:400">__SET__</span></h1>
  <div class="prog"><div id="bar"></div></div>
  <span class="progtxt" id="ptxt">0 / 0</span>
</header>

<main>
  <div class="imgwrap">
    <div class="idbar"><span class="iid" id="iid">—</span><span class="flag" id="flag"></span></div>
    <img id="img" alt="tongue">
    <div class="grid-jump" id="jump"></div>
  </div>
  <div>
    <div class="fields" id="fields"></div>
    <nav>
      <button id="prev">‹ Prev</button>
      <button id="next">Next ›</button>
      <span class="spacer"></span>
      <span class="hint">keys: ←/→ nav</span>
    </nav>
  </div>
</main>

<div class="foot">
  <button class="primary" id="export">⬇ Export JSON</button>
  <button id="importbtn">⬆ Import / resume</button>
  <input type="file" id="importfile" accept="application/json" style="display:none">
  <button id="clear">Clear all</button>
  <span class="hint" id="saved">autosaves to this browser</span>
</div>

<script>
const IMAGES = __IMAGES__;
const FLAGS = __FLAGS__;
const IDS = Object.keys(IMAGES);
const FIELDS = __FIELDS__;
const LS_KEY = "__LS_KEY__";
let data = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
let cur = 0;

const $ = id => document.getElementById(id);
function isDone(id){ const d=data[id]; return d && FIELDS.every(f=>d[f.key]); }

function render(){
  const id = IDS[cur];
  $("iid").textContent = id;
  $("flag").textContent = FLAGS[id] ? "⚑ "+FLAGS[id] : "";
  $("img").src = IMAGES[id] || "";
  const wrap = $("fields"); wrap.innerHTML = "";
  const d = data[id] || (data[id]={});
  for(const f of FIELDS){
    const div = document.createElement("div"); div.className="field";
    div.innerHTML = `<div class="lab">${f.lab}</div><div class="help">${f.help}</div>`;
    const opts = document.createElement("div"); opts.className="opts";
    f.opts.forEach(o=>{
      const b=document.createElement("button"); b.textContent=o;
      if(d[f.key]===o) b.classList.add("on");
      b.onclick=()=>{ d[f.key]=o; save(); render(); };
      opts.appendChild(b);
    });
    div.appendChild(opts); wrap.appendChild(div);
  }
  const jg=$("jump"); jg.innerHTML="";
  IDS.forEach((id2,i)=>{
    const b=document.createElement("button"); b.textContent=id2.slice(1);
    if(isDone(id2)) b.classList.add("done");
    if(i===cur) b.classList.add("cur");
    b.onclick=()=>{ cur=i; render(); };
    jg.appendChild(b);
  });
  const done = IDS.filter(isDone).length;
  $("bar").style.width = (100*done/IDS.length)+"%";
  $("ptxt").textContent = `${done} / ${IDS.length}`;
}
function save(){ localStorage.setItem(LS_KEY, JSON.stringify(data)); }
function go(d){ cur=(cur+d+IDS.length)%IDS.length; render(); }

$("prev").onclick=()=>go(-1);
$("next").onclick=()=>go(1);
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT")return;
  if(e.key==="ArrowLeft")go(-1);
  else if(e.key==="ArrowRight")go(1);
});
$("export").onclick=()=>{
  const out={};
  for(const id of IDS){ if(data[id] && Object.keys(data[id]).length) out[id]=data[id]; }
  const blob=new Blob([JSON.stringify(out,null,1)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="__OUTNAME__"; a.click();
};
$("importbtn").onclick=()=>$("importfile").click();
$("importfile").onchange=e=>{
  const f=e.target.files[0]; if(!f)return;
  const r=new FileReader();
  r.onload=()=>{ try{ const j=JSON.parse(r.result); delete j._legend;
    Object.assign(data,j); save(); render(); }catch(err){ alert("bad JSON: "+err); } };
  r.readAsText(f);
};
$("clear").onclick=()=>{ if(confirm("Clear all labels in this browser?")){ data={}; save(); render(); } };
render();
</script>
</body>
</html>
"""

html = (HTML.replace("__IMAGES__", img_js)
            .replace("__FLAGS__", json.dumps(FLAGS))
            .replace("__FIELDS__", json.dumps(FIELDS))
            .replace("__LS_KEY__", LS_KEY)
            .replace("__OUTNAME__", f"{SET}_{args.mode}_labels.json")
            .replace("__SET__", f"{SET} · {args.mode}"))
with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT}  ({len(html)/1e6:.1f} MB, {len(images)} images, mode={args.mode}, {len(FIELDS)} fields)")
