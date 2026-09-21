from xenpipe.normalize import fingerprint_text, normalize_text_record


def test_text_normalization():
    instruction, response = normalize_text_record({"instruction": "  hello  ", "response": " world "})
    assert instruction == "hello"
    assert response == "world"


def test_fingerprint_is_stable():
    assert fingerprint_text("abc") == fingerprint_text("abc")
    assert fingerprint_text("abc") != fingerprint_text("abcd")
