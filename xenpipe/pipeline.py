from dataclasses import asdict
from .streaming import authorized_stream
from .normalize import *

TARGETS={"gen1-t":"text","gen1-t-code":"text","gen1-t-fast":"text","gen1-t-plan":"text","gen1-i":"image","gen1-v":"video"}

def iter_xen_samples(dataset_id,*,target,split="train",revision=None,config=None,token=None,text_column=None,response_column=None,media_column=None,caption_column=None,max_samples=None,strict_license=True,allowed_licenses=None,deduplicate=True,image_size=None,cache_dir=None):
    if target not in TARGETS: raise ValueError(f"Unknown target: {target}")
    ds,base=authorized_stream(dataset_id,split,revision,config,token,strict_license,allowed_licenses)
    seen=set(); count=0
    for row in ds:
        if TARGETS[target]=="text":
            instruction,response=normalize_text_record(row,text_column,response_column)
            if not instruction: continue
            key=fingerprint_text(instruction+"\n"+(response or ""))
            if deduplicate and key in seen: continue
            seen.add(key); p=asdict(base); p["target"]=target
            yield {"type":"text","instruction":instruction,"response":response,"fingerprint":key,"provenance":p}
        elif target=="gen1-i":
            col=media_column or "image"
            if col not in row: continue
            caption=normalize_caption(row,caption_column)
            if not caption: continue
            try: payload=image_rgb_bytes(row[col],image_size)
            except Exception: continue
            key=fingerprint_bytes(payload)
            if deduplicate and key in seen: continue
            seen.add(key); p=asdict(base); p["target"]=target
            yield {"type":"image","image":payload,"caption":caption,"fingerprint":key,"provenance":p}
        else:
            col=media_column or "video"
            if col not in row: continue
            caption=normalize_caption(row,caption_column)
            if not caption: continue
            try: video=resolve_video(row[col],dataset_id,revision,cache_dir)
            except Exception: continue
            key=fingerprint_text(f"{dataset_id}|{base.revision}|{count}|{row.get(col)}")
            if deduplicate and key in seen: continue
            seen.add(key); p=asdict(base); p["target"]=target
            yield {"type":"video","video":video,"caption":caption,"fingerprint":key,"provenance":p}
        count+=1
        if max_samples is not None and count>=max_samples: break
