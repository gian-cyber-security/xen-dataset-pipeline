import hashlib, io, json, re
from PIL import Image

TEXT_COLUMNS=("text","content","document","prompt","instruction")
RESPONSE_COLUMNS=("response","answer","output","completion")
CAPTION_COLUMNS=("caption","text","description","prompt")

def clean_text(value):
    if value is None: return ""
    if isinstance(value,(dict,list)): value=json.dumps(value,ensure_ascii=False)
    return re.sub(r"\\s+"," ",str(value).replace("\\x00"," ")).strip()

def first_value(row, columns):
    for c in columns:
        if row.get(c) not in (None,""): return row[c]
    return None

def normalize_text_record(row,text_column=None,response_column=None):
    instruction=clean_text(row.get(text_column)) if text_column else clean_text(first_value(row,TEXT_COLUMNS))
    response=clean_text(row.get(response_column)) if response_column else clean_text(first_value(row,RESPONSE_COLUMNS))
    return instruction,response or None

def normalize_caption(row,column=None):
    return clean_text(row.get(column)) if column else clean_text(first_value(row,CAPTION_COLUMNS))

def image_bytes(value):
    if isinstance(value,bytes): return value
    if isinstance(value,dict):
        if value.get("bytes") is not None: return bytes(value["bytes"])
        if value.get("path"): return open(value["path"],"rb").read()
    if isinstance(value,Image.Image):
        b=io.BytesIO(); value.convert("RGB").save(b,format="PNG"); return b.getvalue()
    if isinstance(value,str): return open(value,"rb").read()
    raise TypeError(f"Unsupported image value: {type(value)!r}")

def image_rgb_bytes(value,size=None):
    with Image.open(io.BytesIO(image_bytes(value))) as img:
        img=img.convert("RGB")
        if size: img.thumbnail((size,size),Image.Resampling.LANCZOS)
        b=io.BytesIO(); img.save(b,format="PNG",optimize=True); return b.getvalue()

def fingerprint_text(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
def fingerprint_bytes(data): return hashlib.sha256(data).hexdigest()
