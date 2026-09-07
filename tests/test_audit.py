from generation.audit import FactAuditor


def test_fact_auditor_clean_answer():
    auditor = FactAuditor()
    evidence = [
        "Table 25: Residential land use has an area of 321.89 sq.km and a proposed share of 55.04%."
    ]
    answer = "The proposed residential land use in PPA is 55.04% covering 321.89 sq.km [1]."
    result = auditor.audit(answer, evidence)

    assert result.is_clean
    assert "55.04%" in result.verified or "55.04" in result.verified
    assert "321.89" in result.verified
    assert len(result.unverified) == 0


def test_fact_auditor_detects_hallucination():
    auditor = FactAuditor()
    evidence = [
        "Table 25: Residential land use proposed share is 55.04%."
    ]
    # Model hallucinated 65.50% and 9999
    answer = "The proposed residential land use is 65.50% with 9999 units [1]."
    result = auditor.audit(answer, evidence)

    assert not result.is_clean
    assert "65.50%" in result.unverified or "65.50" in result.unverified
    assert "9999" in result.unverified
