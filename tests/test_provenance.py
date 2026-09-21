from xenpipe.provenance import DatasetProvenance


def test_provenance_serializes():
    p = DatasetProvenance("huggingface","example/test","abc","cc0-1.0","https://huggingface.co/datasets/example/test","2026-01-01T00:00:00+00:00","gen1-t","strict")
    assert p.to_dict()["target"] == "gen1-t"
