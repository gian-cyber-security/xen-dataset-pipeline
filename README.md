# XEN Dataset Pipeline

Automated, provenance-aware dataset pipeline built specifically for the XEN AI model family.

This repository is the dataset infrastructure for XEN-GEN1-T, XEN-GEN1-T-Code, XEN-GEN1-T-Fast, XEN-GEN1-T-Plan, XEN-GEN1-I, and XEN-GEN1-V. It streams real datasets, validates provenance and declared licensing metadata, normalizes samples, deduplicates them, and exposes XEN-ready samples directly to training code.

## Core flow

```text
Real dataset source
      |
      v
License + provenance gate
      |
      v
Streaming loader
      |
      v
Quality / normalization
      |
      v
SHA-256 deduplication
      |
      v
XEN target adapter
      |
      +--> GEN1-T / T-Code / T-Fast / T-Plan
      +--> GEN1-I
      +--> GEN1-V
      |
      v
Direct trainer iterator
```

The core is Python because Hugging Face Datasets, image decoding, video decoding, and ML training have mature Python APIs. The architecture is not language-locked; performance-critical components can later be replaced by Rust/C++ while preserving the XEN sample contract.

## Real data only

The pipeline does not generate fake/mock training examples and does not bundle a third-party training corpus.

The primary source integration is Hugging Face Datasets. Hugging Face supports streaming large datasets without downloading the entire dataset first, and its video feature can expose decoded video objects and frames.

A dataset being hosted on Hugging Face is **not automatically copyright-free**. Users must inspect the dataset card, license, terms, and applicable law for every dataset they select.

The pipeline records platform, dataset ID, revision, declared license, source URL, processing timestamp, XEN target, policy mode, and sample fingerprint.

Strict mode rejects missing or unapproved declared licenses. This automated check is not legal advice and cannot determine ownership of every individual item.

## XEN targets

| Target | Expected data |
|---|---|
| `gen1-t` | general text / instruction-response |
| `gen1-t-code` | code / coding instruction-response |
| `gen1-t-fast` | short/simple instruction-response |
| `gen1-t-plan` | planning / structured tasks |
| `gen1-i` | image + caption |
| `gen1-v` | video + caption |

## Setup

Requirements: Python 3.11+, Git, and FFmpeg for video workflows.

```bash
git clone https://github.com/gian-cyber-security/xen-dataset-pipeline.git
cd xen-dataset-pipeline
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
```

Install:

```bash
pip install -e .
```

For video decoding:

```bash
pip install -e ".[video]"
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Recommended video starting point

For a first GEN1-V pipeline test, Hugging Face `nkp37/OpenVid-1M` exposes `video` and `caption` fields and the inspected revision can be recorded by the pipeline. Its declared dataset license is CC BY 4.0. Its dataset card also says the videos were collected from other public datasets, so users must still follow the original-source licenses and terms; the Hub declaration is not blanket legal clearance.

Fast metadata inspection (does **not** download video files):

```bash
xenpipe inspect-dataset --dataset nkp37/OpenVid-1M
xenpipe stream --dataset nkp37/OpenVid-1M --target gen1-v --media-column video --caption-column caption --max-samples 3 --show 3
```

If you explicitly want to resolve/download the three videos:

```bash
xenpipe stream --dataset nkp37/OpenVid-1M --target gen1-v --media-column video --caption-column caption --max-samples 3 --show 3 --resolve-media
```

### OpenVid-1M large-ZIP optimization

OpenVid-1M is special: its `video` column can contain a filename that lives inside an `OpenVidHD_part_*.zip`, rather than a standalone file at the dataset root. For this dataset, GEN1-V resolution uses a ZIP64-aware HTTP Range extractor instead of trying to download the multi-gigabyte archive.

The resolver:
- downloads the small `OpenVidHD.json` index and builds a filename-to-part map
- downloads each part's ZIP central directory once into the XEN cache
- uses HTTP Range requests to fetch only the requested ZIP member
- supports ZIP64 metadata and Deflate-compressed members
- validates the extracted bytes with the declared size and CRC32
- stores only the requested MP4 in the temporary cache

For example, the tested OpenVid video `---_iRTHryQ_13_0to241.mp4` is inside part 8. Its ZIP member is about 4 MB, so the resolver can fetch the member without downloading the roughly 46.5 GiB part-8 archive.

This optimization is specific to OpenVid's archive layout. Other datasets continue to use the normal Hugging Face video resolver.

### GEN1-V performance

GEN1-V training uses **lazy video resolution + bounded parallel prefetch**. The dataset iterator first produces lightweight references, then a small background worker pool downloads only the upcoming samples. This avoids the old behavior where merely inspecting three samples could block on remote video downloads.

Defaults:
- 4 background download workers
- 8 samples in flight
- temporary Hugging Face/XEN cache
- ordered delivery to the trainer
- no permanent processed dataset is created

Tune the workers/window for your machine:

```bash
export XEN_PREFETCH_WORKERS=6
export XEN_PREFETCH_SIZE=12
```

For Colab, start around 4 workers / 8 prefetched samples. Increasing them does **not** guarantee faster training; network bandwidth and HF server throughput can become the bottleneck.

For direct training, use `xenpipe.training.iter_training_samples` or `stream_to_trainer`. The training adapter resolves video on demand and prefetches in the background.

## Inspect a dataset

```bash
xenpipe inspect-dataset --dataset owner/dataset
```

Strict mode is the default. If a dataset has no declared license or its license is outside the current allow-list, processing stops.

## Direct XEN training

The preferred mode does not create a permanent processed dataset file. XEN trainers consume samples directly:

```python
from xenpipe.training import stream_to_trainer


def train_one(sample):
    # Convert sample to tensors and run one XEN training update.
    # sample also contains provenance and a fingerprint.
    pass

stream_to_trainer(
    train_one,
    "owner/dataset",
    target="gen1-t",
    split="train",
    text_column="text",
    max_samples=10000,
)
```

The same API works for `gen1-t-code`, `gen1-t-fast`, `gen1-t-plan`, `gen1-i`, and `gen1-v`. This lets each XEN model repository own its training loop while this repository owns data acquisition, filtering, provenance, and modality adaptation.

Network/decoder caches may still exist temporarily. They are not a permanent training corpus and can be removed with `xenpipe clean-cache` when using the CLI cache directory.

## Optional export

Offline export is available for reproducible experiments:

```bash
xenpipe export --dataset owner/dataset --target gen1-t --text-column text --output datasets/gen1-t.jsonl
```

Image export:

```bash
xenpipe export --dataset owner/dataset --target gen1-i --media-column image --caption-column caption --output datasets/gen1-i.jsonl
```

Direct video mode is preferred. The JSONL exporter intentionally refuses to silently serialize decoded video objects into an opaque permanent corpus.

## Normalization

Text adapters recognize common fields such as `text`, `content`, `document`, `prompt`, `instruction`, plus `response`, `answer`, `output`, and `completion`.

Image adapters accept Hugging Face image features, bytes, paths, and PIL images and normalize them to RGB PNG bytes.

Video adapters preserve the decoded video object in direct mode so the XEN-V trainer can decode only what it needs.

## Deduplication

Text and normalized image samples use SHA-256 fingerprints. Video direct mode uses a stream-position fingerprint so the pipeline does not force a complete remote video download merely to hash it. A future perceptual-hash module can add content-level similarity detection.

## Cache

Default cache directory:

```text
.xen-cache/
```

Clean it with:

```bash
xenpipe clean-cache
```

## Development

GitHub Actions runs the tests on pushes and pull requests.

The repository is intentionally modular so future Rust/C++ accelerators can replace bottleneck components without changing the public XEN sample contract.

## Repository boundary

This repository contains the **XEN data pipeline**, not model weights and not a third-party dataset mirror. The XEN model repositories remain separate.

Third-party datasets keep their original licenses and terms. The Apache-2.0 license of this source repository does not relicense third-party data.

## License

The XEN Dataset Pipeline source code is licensed under the Apache License 2.0.
