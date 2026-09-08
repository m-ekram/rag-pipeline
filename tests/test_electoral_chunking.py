"""Tests for electoral roll parsing and record-boundary chunking."""

import pytest
from ingestion.documents import Document
from ingestion.electoral import (
    is_electoral_text,
    parse_electoral_records,
    VoterRecord,
)
from ingestion.chunking import (
    ElectoralRecordChunker,
    SentenceAwareChunker,
)

SAMPLE_ELECTORAL_TEXT = """
विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम : 183-कुम्हरार
भाग संख्या : : 7
अनुभाग संख्या एवं नाम : 1-एनी वेसेन्ट
101]
SHS1234567
102]
SHS7654321
103]
JDK9988776
निर्वाचक का नाम : राहुल कुमार
निर्वाचक का नाम : प्रिया शर्मा
निर्वाचिक का नाम : अमित वर्मा
पिता का नाम: विक्रम कुमार
पति का नाम: रोहित शर्मा
पिता का नाम: सुरेश वर्मा
मकान संख्या : 12
मकान संख्या : 14/A
मकान संख्या : 18
फोटो उपलब्ध
उम्र: 28 लिंग: : पुरुष
उम्र: 26 लिंग: : महिला
उम्र: 34 लिंग: : पुरुष
"""

SAMPLE_PLAIN_TEXT = """
A Roth IRA is an individual retirement account that offers tax-free growth.
Contributions are made with after-tax dollars, meaning you cannot deduct them on your tax return.
However, withdrawals in retirement are completely tax-free.
"""


def test_is_electoral_text():
    assert is_electoral_text(SAMPLE_ELECTORAL_TEXT) is True
    assert is_electoral_text(SAMPLE_PLAIN_TEXT) is False
    assert is_electoral_text("") is False


def test_parse_electoral_records():
    header, records = parse_electoral_records(SAMPLE_ELECTORAL_TEXT)
    assert "183-कुम्हरार" in header
    assert "भाग संख्या" in header
    assert len(records) == 3

    # Verify first voter
    r1 = records[0]
    assert r1.serial == "101"
    assert r1.epic == "SHS1234567"
    assert r1.name == "राहुल कुमार"
    assert "विक्रम कुमार" in r1.relation
    assert r1.house == "12"
    assert r1.age == "28"
    assert r1.gender == "पुरुष"

    md = r1.to_markdown()
    assert "Serial: 101" in md
    assert "EPIC: SHS1234567" in md
    assert "Voter: राहुल कुमार" in md

    # Verify second voter
    r2 = records[1]
    assert r2.serial == "102"
    assert r2.epic == "SHS7654321"
    assert r2.name == "प्रिया शर्मा"
    assert "पति: रोहित शर्मा" in r2.relation


def test_electoral_chunker_packs_records_atomically():
    doc = Document(
        doc_id="test_eroll#p3",
        text=SAMPLE_ELECTORAL_TEXT,
        title="Test E-Roll",
        page=3,
    )
    chunker = ElectoralRecordChunker(records_per_chunk=2, group_by_household=False)
    chunks = list(chunker.chunk(doc))

    assert len(chunks) == 2  # 3 records with batch size 2 -> 2 chunks
    assert chunks[0].page == 3
    assert chunks[0].doc_id == "test_eroll#p3"
    assert chunks[0].metadata["chunk_type"] == "electoral_records"
    assert chunks[0].metadata["voter_count"] == 2

    # Chunk 0 contains voter 101 and 102
    assert "SHS1234567" in chunks[0].text
    assert "SHS7654321" in chunks[0].text
    assert "SHS1234567" not in chunks[1].text

    # Chunk 1 contains voter 103
    assert "JDK9988776" in chunks[1].text
    assert chunks[1].metadata["voter_count"] == 1


def test_electoral_chunker_fallback_on_plain_text():
    doc = Document(
        doc_id="plain_doc#p1",
        text=SAMPLE_PLAIN_TEXT,
        title="Plain Doc",
        page=1,
    )
    chunker = ElectoralRecordChunker(records_per_chunk=5)
    chunks = list(chunker.chunk(doc))

    assert len(chunks) >= 1
    # Fallback should not tag it as electoral_records
    assert chunks[0].metadata.get("chunk_type") != "electoral_records"
    assert "Roth IRA" in chunks[0].text


def test_electoral_chunker_page1_metadata():
    page1_text = """
निर्वाचक नामावली 2025 S04 बिहार
विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम : 183-कुम्हरार
भाग संख्या : 7
मतदान केंद्र संख्या एवं नाम : 7 - रा० प्रा० वि० लोहिया नगर (पूर्वी भाग)
मतदाताओं की कुल संख्या : 950
"""
    doc = Document(
        doc_id="2025-EROLLGEN-S04-183#p1",
        text=page1_text,
        title="183-Kumhrar E-Roll",
        page=1,
    )
    chunker = ElectoralRecordChunker(records_per_chunk=5)
    chunks = list(chunker.chunk(doc))

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "2025-EROLLGEN-S04-183#p1::metadata"
    assert "### निर्वाचन नामावली एवं मतदान केंद्र विवरण (Polling Station Metadata)" in c.text
    assert "183-कुम्हरार" in c.text
    assert "मतदान केंद्र संख्या एवं नाम" in c.text
    assert c.metadata.get("block_type") == "electoral_metadata"
    assert c.metadata.get("page_num") == 1


SAMPLE_HOUSEHOLD_TEXT = """
विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम : 183-कुम्हरार
भाग संख्या : 7
अनुभाग संख्या एवं नाम : 1-एनी वेसेन्ट
241]
BR/35/207/282142
242]
BR/35/207/284016
243]
JDK2337657
244]
SHS1104165
245]
SHS0947168
निर्वाचक का नाम : किरण देवी
निर्वाचक का नाम : विजय प्रसाद
निर्वाचक का नाम : तनवीर अहमद
निर्वाचक का नाम : मनीषा
निर्वाचक का नाम : काशिफ अहमद
पति का नाम: रामजी प्रसाद
पिता का नाम: शीतल प्रसाद
पिता का नाम: स्वः शमीम अहमद
पिता का नाम: श्री शीतल प्रसाद
पिता का नाम: स्वः शमीम अहमद
मकान संख्या : 36
मकान संख्या : 36
मकान संख्या : 36
मकान संख्या : 36
मकान संख्या : 36
उम्र: 57 लिंग: : महिला
उम्र: 52 लिंग: : पुरुष
उम्र: 42 लिंग: : पुरुष
उम्र: 41 लिंग: : महिला
उम्र: 40 लिंग: : पुरुष
"""


def test_electoral_chunker_household_co_location():
    doc = Document(
        doc_id="test_eroll#p11",
        text=SAMPLE_HOUSEHOLD_TEXT,
        title="Test E-Roll Page 11",
        page=11,
    )
    chunker = ElectoralRecordChunker()  # Default group_by_household=True
    chunks = list(chunker.chunk(doc))

    # 5 voters in House 36 -> 5 child chunks
    assert len(chunks) == 5

    # Check child chunk 2 (Tanveer Ahmad, serial 243)
    tanveer_chunk = next(c for c in chunks if c.metadata.get("serial") == "243")
    assert "तनवीर अहमद" in tanveer_chunk.text
    assert tanveer_chunk.metadata["house_num"] == "36"
    assert tanveer_chunk.metadata["parent_id"] == "test_eroll#p11#house_36"

    # Verify parent text has all 5 family members
    parent_text = tanveer_chunk.metadata["parent_text"]
    assert "### परिवार / मकान संख्या: 36 (कुल सदस्य: 5)" in parent_text
    assert "किरण देवी" in parent_text
    assert "तनवीर अहमद" in parent_text
    assert "काशिफ अहमद" in parent_text
    assert "स्वः शमीम अहमद" in parent_text

    # Verify prompt expansion deduplication works as expected
    from generation.prompts import _as_chunks
    expanded = _as_chunks(chunks)
    # Even though 5 child chunks from House 36 were provided, they collapse into 1 parent household block!
    assert len(expanded) == 1
    assert "### परिवार / मकान संख्या: 36 (कुल सदस्य: 5)" in expanded[0].text


def test_house_number_cleaning_from_ocr_artifacts():
    from ingestion.chunking import canonicalize_house_number, dual_house_number_display
    from ingestion.electoral import HOUSE_NO_PATTERN

    # 1. Test regex boundary stops before फोटो उपलब्ध and pipes
    line = "मकान संख्या : 3 फोटो उपलब्ध | | | मकान संख्या: 3 फोटो उपलब्ध"
    matches = HOUSE_NO_PATTERN.findall(line)
    assert matches == ["3", "3"]

    # 2. Test canonicalize_house_number removes trailing noise
    assert canonicalize_house_number("3 फोटो उपलब्ध | | | मकान संख्या: 3 फोटो उपलब्ध") == "3"
    assert canonicalize_house_number("३ फोटो उपलब्ध | | |") == "3"
    assert canonicalize_house_number("14/A फोटो") == "14/A"
    assert canonicalize_house_number("") == ""

    # 3. Test dual display
    assert dual_house_number_display("3 फोटो उपलब्ध") == "3 / ३"
def test_linearize_electoral_summary_table():
    from ingestion.chunking import linearize_electoral_summary_table

    # Test 1: Booth 25 real OCR text with Devanagari danda '।' for 1 and '2022' for 1022
    raw_ocr = """
    विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम : 183-कुम्हरार
    भाग संख्या : 25
    मतदाताओं की संख्या :
    आरंभिक क्रम अंतिम क्रम मतदाताओं की संख्या
    संख्या संख्या पुरुष महिला तृतीय लिंग कुल
    । 703 528 494 0 2022
    """
    linearized = linearize_electoral_summary_table(raw_ocr)
    assert "### [सारणीबद्ध सारांश रिकॉर्ड / Labeled Summary Record]" in linearized
    assert "आरंभिक क्रम संख्या: 1" in linearized
    assert "अंतिम क्रम संख्या: 703" in linearized
    assert "पुरुष मतदाताओं की कुल संख्या: 528" in linearized
    assert "महिला मतदाताओं की कुल संख्या: 494" in linearized
    assert "तृतीय लिंग मतदाताओं की संख्या: 0" in linearized
    # Self-consistency check must fix 2022 to 1022 (528 + 494)
    assert "मतदाताओं की कुल संख्या (Total Voters): 1022" in linearized

    # Test 2: Devanagari digits
    devanagari_ocr = """
    मतदाताओं की संख्या
    आरंभिक क्रम अंतिम क्रम पुरुष महिला तृतीय लिंग कुल
    १ १०३१ ५२८ ४९४ ० १०२२
    """
    dev_lin = linearize_electoral_summary_table(devanagari_ocr)
    assert "पुरुष मतदाताओं की कुल संख्या: 528" in dev_lin
    assert "महिला मतदाताओं की कुल संख्या: 494" in dev_lin
    assert "मतदाताओं की कुल संख्या (Total Voters): 1022" in dev_lin

    # Test 3: Plain text without summary keywords returns unchanged
    plain = "This is a random document text with no electoral summary."
    assert linearize_electoral_summary_table(plain) == plain


def test_multi_column_row_parsing_kashish_kashyap():
    page4_snippet = """
विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम : 183-कुम्हरार
भाग संख्या : : 61
अनुभाग संख्या एवं नाम : 1-राजेंद्र नगर
‘ | 34 | 8094926432...। | 35| 8084926440... ] 36| SHS5I24394
निर्वाचक का नाम : अजय कुमार गुप्ता हु निर्वाचक का नाम : श्रेष्ठा राज हु निर्वाचक का नाम: कशिश कश्यप .
पिता का नाम:: अखिला नंद प्रसाद पिता का नाम:: अजय कुमार गुप्ता पति का नाम: अनिमेश कश्यप
मकान संख्या : 402 फोटो उपलब्ध... मकान संयम: 402 फोटो उपलब्ध... मकान संकया : 402-403,हैम छठ धाम जपार्टमेंट. फोदो उपलब्ध
उप्र : 63 लिंग: : पुरुष Wa: 27 लिंग: : महिला उम्र ; 23 लिंग; : महिला
"""
    header, records = parse_electoral_records(page4_snippet)
    assert len(records) == 3

    # Voter 3 is Kashish Kashyap
    r3 = records[2]
    assert r3.serial == "36"
    assert r3.epic == "SHS5124394"  # Normalized 'I' to '1'
    assert r3.name == "कशिश कश्यप"
    assert "पति: अनिमेश कश्यप" in r3.relation
    assert "402-403" in r3.house
    assert r3.age == "23"
    assert "महिला" in r3.gender
    assert "कशिश कश्यप (Kashish Kashyap)" in r3.to_markdown()


def test_transliteration_dual_script():
    v = VoterRecord(serial="10", epic="SHS4590493", name="फ़राज़ अहमद", house="4", age="28", gender="पुरुष")
    md = v.to_markdown()
    assert "Voter: फ़राज़ अहमद (Faraz Ahmad)" in md

