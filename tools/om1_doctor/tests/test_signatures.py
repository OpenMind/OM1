from om1_doctor.checks.signatures import match_signatures

def test_permission_denied_signature():
    text = "Error: Permission denied while accessing file"
    results = match_signatures(text)
    assert any(r["id"] == "PERM_DENIED" for r in results)

def test_no_match():
    text = "Everything is running fine"
    results = match_signatures(text)
    assert results == []
