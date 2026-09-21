import base64,json
from pathlib import Path
from .pipeline import iter_xen_samples

def export_jsonl(output,dataset_id,target,**kwargs):
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); n=0
    with path.open("w",encoding="utf-8") as f:
        for s in iter_xen_samples(dataset_id,target=target,**kwargs):
            s=dict(s)
            if isinstance(s.get("image"),bytes): s["image"]={"encoding":"base64","data":base64.b64encode(s.pop("image")).decode()}
            if s.get("type")=="video": raise ValueError("Video JSONL export is disabled; use direct streaming to avoid permanent media copies")
            f.write(json.dumps(s,ensure_ascii=False,separators=(",",":"))+"\\n"); n+=1
    return n
