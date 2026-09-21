from datasets import load_dataset
from .license import inspect_hf_dataset,check_license
from .provenance import DatasetProvenance,now_iso

def authorized_stream(dataset_id,split="train",revision=None,config=None,token=None,strict_license=True,allowed_licenses=None):
    info=inspect_hf_dataset(dataset_id,revision)
    check_license(info,strict_license,allowed_licenses)
    kwargs={"path":dataset_id,"split":split,"streaming":True,"revision":revision,"token":token}
    if config: kwargs["name"]=config
    ds=load_dataset(**kwargs)
    prov=DatasetProvenance("huggingface",dataset_id,info.revision,info.license,info.source_url,now_iso(),"unknown","strict" if strict_license else "permissive")
    return ds,prov
