import argparse,json,os,shutil
from pathlib import Path
from .license import inspect_hf_dataset,check_license,DEFAULT_ALLOWED
from .pipeline import iter_xen_samples
from .export import export_jsonl

def main():
    p=argparse.ArgumentParser(prog="xenpipe"); sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("inspect-dataset"); q.add_argument("--dataset",required=True); q.add_argument("--revision"); q.add_argument("--permissive",action="store_true")
    def common(x):
        x.add_argument("--dataset",required=True); x.add_argument("--target",required=True,choices=["gen1-t","gen1-t-code","gen1-t-fast","gen1-t-plan","gen1-i","gen1-v"]); x.add_argument("--split",default="train"); x.add_argument("--revision"); x.add_argument("--config"); x.add_argument("--text-column"); x.add_argument("--response-column"); x.add_argument("--media-column"); x.add_argument("--caption-column"); x.add_argument("--max-samples",type=int); x.add_argument("--permissive",action="store_true"); x.add_argument("--no-dedup",action="store_true"); x.add_argument("--image-size",type=int); x.add_argument("--cache-dir",default=os.getenv("XEN_CACHE_DIR",".xen-cache"))
    s=sub.add_parser("stream"); common(s); s.add_argument("--show",type=int,default=3); s.add_argument("--resolve-media",action="store_true",help="Resolve/download media while streaming; off by default for fast metadata inspection.")
    e=sub.add_parser("export"); common(e); e.add_argument("--output",required=True)
    c=sub.add_parser("clean-cache"); c.add_argument("--cache-dir",default=os.getenv("XEN_CACHE_DIR",".xen-cache"))
    a=p.parse_args()
    if a.cmd=="inspect-dataset":
        i=inspect_hf_dataset(a.dataset,a.revision); check_license(i,not a.permissive); print(json.dumps(i.__dict__,indent=2)); return
    if a.cmd=="clean-cache":
        x=Path(a.cache_dir)
        if x.exists(): shutil.rmtree(x)
        print(f"cache cleaned: {x}"); return
    kw=dict(split=a.split,revision=a.revision,config=a.config,text_column=a.text_column,response_column=a.response_column,media_column=a.media_column,caption_column=a.caption_column,max_samples=a.max_samples,strict_license=not a.permissive,allowed_licenses=DEFAULT_ALLOWED,deduplicate=not a.no_dedup,image_size=a.image_size,cache_dir=a.cache_dir)
    if a.cmd=="stream":
        kw["resolve_media"]=a.resolve_media
        for i,s in enumerate(iter_xen_samples(a.dataset,target=a.target,**kw)):
            if i>=a.show: break
            s=dict(s)
            if isinstance(s.get("image"),bytes): s["image"]=f"<{len(s['image'])} bytes>"
            if s.get("type")=="video" and "video" in s: s["video"]="<resolved video>"
            if s.get("type")=="video" and "video_ref" in s: s["video_ref"]=str(s["video_ref"])[:500]
            print(json.dumps(s,ensure_ascii=False,default=str,indent=2))
    else:
        print(f"exported {export_jsonl(a.output,a.dataset,a.target,**kw)} samples")

if __name__=="__main__": main()
