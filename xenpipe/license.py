from dataclasses import dataclass
from huggingface_hub import HfApi

DEFAULT_ALLOWED = {"cc0-1.0","cc-by-4.0","cc-by-3.0","cc-by-2.0","unlicense","mit","bsd-2-clause","bsd-3-clause","apache-2.0"}

@dataclass(frozen=True)
class LicenseInfo:
    dataset_id: str
    license: str | None
    revision: str | None
    source_url: str

    @property
    def normalized(self):
        return self.license.strip().lower().replace("_","-") if self.license else None

def inspect_hf_dataset(dataset_id, revision=None):
    info = HfApi().dataset_info(dataset_id, revision=revision)
    card = info.card_data
    value = None
    if card is not None:
        try: value = card.get("license")
        except AttributeError: value = getattr(card, "license", None)
    if isinstance(value, list): value = value[0] if value else None
    return LicenseInfo(dataset_id, str(value) if value else None, getattr(info,"sha",None) or revision, f"https://huggingface.co/datasets/{dataset_id}")

def check_license(info, strict=True, allowed=None):
    allowed = allowed or DEFAULT_ALLOWED
    if info.normalized is None:
        if strict: raise PermissionError(f"Dataset {info.dataset_id} has no declared license")
        return
    if strict and info.normalized not in allowed:
        raise PermissionError(f"Dataset {info.dataset_id} declares '{info.license}', not allowed by current policy")
