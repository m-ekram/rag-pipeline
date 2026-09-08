"""Electoral roll table reconstruction and record extraction.

Electoral roll scans are laid out in a 3-column grid of voter cards.
PaddleOCR's top-to-bottom sorting reads each field category across columns:
- All serial numbers and EPIC IDs across the row are read first.
- All voter names are read next.
- All relationships (father/husband/mother) are read next.
- All house numbers are read next.
- All ages and genders are read next.

This module re-aligns those parallel streams into complete, atomic voter
records formatted as key-value pairs for reliable LLM comprehension.
"""

from dataclasses import dataclass
import re
from typing import Optional


EPIC_RE = re.compile(
    r"\b([A-Z]{2,4}[0-9]{6,8}|[A-Z]{2}/[0-9]{2}/[0-9]{2,4}/[0-9]{4,8})\b"
)
SERIAL_RE = re.compile(r"^\s*\[?\s*([0-9]{1,4})\s*\]?\s*$")
NAME_RE = re.compile(
    r"^(?:नि[र्वा]+[च|ं|ि|क|्]+\s*का\s*नाम)\s*[:：]?\s*(.*)$"
)
REL_RE = re.compile(
    r"^(?:(पिता|पति|पतिका|माता|अन्य)(?:\s*का)?\s*नाम)\s*[:：]?\s*(.*)$"
)
HOUSE_NO_PATTERN = re.compile(
    r"मकान\s*(?:संख्या|नं|क्र|नम्बर|सं\.)\s*[:\-：]?\s*([०-९0-9A-Za-z\/\-\s]+?)(?=\s*(?:फोटो|उपलब्ध|हस्ताक्षर|आयु|लिंग|[|,\n]|$))",
    re.IGNORECASE | re.UNICODE,
)
HOUSE_RE = HOUSE_NO_PATTERN
AGE_GEN_RE = re.compile(
    r"(?:उ[^\d०-९]*?)\s*([0-9०-९]+)\s*.*?(महिला|पुरु[ष|थ|स]|अन्य)",
    re.IGNORECASE,
)


@dataclass
class VoterRecord:
    """Atomic representation of a single voter in an electoral roll."""

    serial: str = ""
    epic: str = ""
    name: str = ""
    relation: str = ""
    house: str = ""
    age: str = ""
    gender: str = ""
    header_context: str = ""

    def to_markdown(self) -> str:
        """Format as structured Key-Value markdown for LLM comprehension."""
        parts = []
        if self.serial:
            parts.append(f"Serial: {self.serial}")
        if self.epic:
            parts.append(f"EPIC: {self.epic}")
        if self.name:
            latin = transliterate_devanagari(self.name)
            if latin and latin.lower() != self.name.lower():
                parts.append(f"Voter: {self.name} ({latin})")
            else:
                parts.append(f"Voter: {self.name}")
        if self.relation:
            parts.append(f"Relation: {self.relation}")
        if self.house:
            clean_house = re.sub(r"(?:फोटो\s*उपलब्ध|फोटो|उपलब्ध|[|:;._])", " ", self.house).strip()
            core_house = clean_house.split()[0] if clean_house.split() else clean_house
            if core_house:
                parts.append(f"House: {core_house}")
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.gender:
            parts.append(f"Gender: {self.gender}")

        return f"- [{ ' | '.join(parts) }]"


def is_electoral_text(text: str) -> bool:
    """Check if the text contains electoral roll structures."""
    if not text:
        return False
    markers = [
        "निर्वाचक",
        "निर्वाचिक",
        "विधानसभा",
        "भाग संख्या",
        "मतदान केंद्र",
        "EPIC",
    ]
    matches = sum(1 for m in markers if m in text)
    return matches >= 2 or len(EPIC_RE.findall(text)) >= 2


def extract_header_context(lines: list[str]) -> str:
    """Extract page-level metadata (constituency, part number, section name)."""
    header_parts = []
    for line in lines:
        if any(k in line for k in ["विधानसभा", "भाग संख्या", "अनुभाग", "लोकसभा"]):
            clean_l = re.sub(r"\s+", " ", line).strip()
            header_parts.append(clean_l)
    return " | ".join(header_parts)


def normalize_epic_id(raw_epic: str) -> str:
    """Normalize OCR character confusions in EPIC numbers (e.g. SHS5I24394 -> SHS5124394)."""
    clean = re.sub(r"[\s|\]\[‘'._]+", "", raw_epic).strip()
    m = re.match(r"^([A-Z]{2,4})(.*)$", clean)
    if m:
        pref, digits = m.groups()
        digits_norm = (
            digits.replace("I", "1")
            .replace("l", "1")
            .replace("|", "1")
            .replace("O", "0")
            .replace("o", "0")
            .replace("D", "0")
            .replace("S", "5")
            .replace("s", "5")
            .replace("B", "8")
            .replace("Z", "2")
        )
        return pref + digits_norm
    return clean


_VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah'
}

_MATRAS = {
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ँ': 'n', 'ः': 'h', '़': ''
}

_CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh',
    'फ़': 'f', 'य़': 'y', 'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy'
}

_NAME_OVERRIDES = {
    "फ़राज़": "Faraz", "फराज": "Faraz", "फेराक": "Faraz",
    "अहमद": "Ahmad", "अहमद्": "Ahmad", "अहमदर": "Ahmad",
    "मोहम्मद": "Mohammad", "मो०": "Md", "मो": "Mohd",
    "कुमार": "Kumar", "कुमारी": "Kumari", "देवी": "Devi",
    "गुप्ता": "Gupta", "शर्मा": "Sharma", "सिंह": "Singh",
    "कश्यप": "Kashyap", "राय": "Rai", "प्रसाद": "Prasad",
    "अजय": "Ajay", "अमित": "Amit", "राहुल": "Rahul",
    "कशिश": "Kashish", "श्रेष्ठा": "Shreshtha", "राज": "Raj",
}

def transliterate_devanagari(text: str) -> str:
    """Convert Hindi Devanagari names into natural Latin phonetics for dual-script indexing."""
    if not text:
        return ""
    words = text.split()
    res_words = []
    for word in words:
        clean_w = word.strip(" ,.:;| हु-")
        if not clean_w:
            continue
        if clean_w in _NAME_OVERRIDES:
            res_words.append(_NAME_OVERRIDES[clean_w])
            continue

        res = []
        i = 0
        n = len(clean_w)
        while i < n:
            if i + 1 < n and clean_w[i:i+2] in _CONSONANTS:
                base = _CONSONANTS[clean_w[i:i+2]]
                i += 2
                if i < n and clean_w[i] in _MATRAS:
                    res.append(base + _MATRAS[clean_w[i]])
                    i += 1
                elif i < n and clean_w[i] == '्':
                    res.append(base)
                    i += 1
                else:
                    res.append(base + ('a' if i < n else ''))
                continue

            ch = clean_w[i]
            if ch in _VOWELS:
                res.append(_VOWELS[ch])
                i += 1
            elif ch in _CONSONANTS:
                base = _CONSONANTS[ch]
                i += 1
                if i < n and clean_w[i] in _MATRAS:
                    res.append(base + _MATRAS[clean_w[i]])
                    i += 1
                elif i < n and clean_w[i] == '्':
                    res.append(base)
                    i += 1
                else:
                    res.append(base + ('a' if i < n else ''))
            elif ch in _MATRAS:
                res.append(_MATRAS[ch])
                i += 1
            elif ch == '्':
                i += 1
            else:
                res.append(ch)
                i += 1
        w_out = "".join(res)
        w_out = re.sub(r'aa$', 'a', w_out)
        if w_out:
            res_words.append(w_out.capitalize())
    return " ".join(res_words)


def parse_electoral_records(text: str) -> tuple[str, list[VoterRecord]]:
    """Parse raw OCR text from a voter grid page into atomic VoterRecords.

    Handles both single-voter vertical streams and multi-voter horizontal row bands.
    Disassembles stacked row lines across 3 columns into clean atomic records.
    """
    if not is_electoral_text(text):
        return "", []

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    header_ctx = extract_header_context(lines)

    # Filter out photo availability placeholders and page headers/footers
    body_lines = [
        l for l in lines
        if not any(l.startswith(ph) for ph in ["फोटो उपलब्ध", "फोटो उपल्ध", "फोटो उपलब"])
        and not any(k in l for k in ["विधानसभा", "भाग संख्या", "अनुभाग", "लोकसभा", "निर्वाचक नामावली 2025", "कुल पृषछ", "प्रकाशन की पूरक"])
    ]

    # Segment lines into rows (each row contains 1-3 voter cards)
    rows: list[list[str]] = []
    cur_row: list[str] = []
    cur_has_data = False

    for l in body_lines:
        m_hdrs = re.findall(r"(?:^|[|\]\[‘'\s])(\d{1,4})\s*[|\]\[:]\s*([^|\]\[\n]+)", l)
        is_serial_line = len(m_hdrs) >= 2 or (len(m_hdrs) == 1 and int(m_hdrs[0][0]) < 2500)
        m_ser = SERIAL_RE.search(l)
        is_single_serial = bool(m_ser and int(m_ser.group(1)) < 2500 and len(m_ser.group(1)) <= 4)
        is_epic = bool(EPIC_RE.search(l))

        if (is_serial_line or is_single_serial or is_epic) and cur_has_data:
            rows.append(cur_row)
            cur_row = []
            cur_has_data = False

        cur_row.append(l)
        if any(k in l for k in ["नाम", "मकान", "लिंग", "उम", "उ्", "उ:", "उम्र"]):
            cur_has_data = True

    if cur_row:
        rows.append(cur_row)

    records: list[VoterRecord] = []
    for r in rows:
        r_serials: list[str] = []
        r_epics: list[str] = []
        r_names: list[str] = []
        r_rels: list[str] = []
        r_houses: list[str] = []
        r_age_gens: list[tuple[str, str]] = []

        for line in r:
            # 1. Card headers (Serial & EPIC)
            m_hdrs = re.findall(r"(?:^|[|\]\[‘'\s])(\d{1,4})\s*[|\]\[:]\s*([^|\]\[\n]+)", line)
            if len(m_hdrs) >= 1 and int(m_hdrs[0][0]) < 2500:
                for s, ep in m_hdrs:
                    r_serials.append(s.strip())
                    clean_ep = normalize_epic_id(ep)
                    if clean_ep:
                        r_epics.append(clean_ep)
                continue

            # Fallback single serial / EPIC
            m_ser = SERIAL_RE.search(line)
            m_ep = EPIC_RE.search(line)
            if m_ep:
                r_epics.append(normalize_epic_id(m_ep.group(1).strip()))
            elif m_ser and int(m_ser.group(1)) < 2500 and len(m_ser.group(1)) <= 4:
                r_serials.append(m_ser.group(1).strip())

            # 2. Multi-name split across columns
            if "नाम" in line and any(k in line for k in ["निर्वाचक", "Prater", "Brae"]):
                splits = [n.strip(" .हु|:：") for n in re.split(r"(?:नि[र्वा]+[च|ं|ि|क|्]+|Prater|Brae)\s*का\s*(?:नाम|ee)\s*[:：]?", line) if n.strip(" .हु|:：")]
                for nm in splits:
                    clean_nm = re.sub(r"^(?:का\s*नाम|नाम)\s*[:：]?\s*", "", nm).strip(" .हु|:：")
                    if clean_nm:
                        r_names.append(clean_nm)
                continue

            # 3. Multi-relation split across columns
            if any(k in line for k in ["पिता", "पति", "माता"]):
                rel_types = re.findall(r"(पिता|पति|पतिका|माता|अन्य)(?:\s*का)?\s*ना[मप्र]+\s*[:：]+", line)
                rel_names = [rn.strip(" .हु|:：") for rn in re.split(r"(?:पिता|पति|पतिका|माता|अन्य)(?:\s*का)?\s*ना[मप्र]+\s*[:：]+", line) if rn.strip(" .हु|:：")]
                if rel_types and rel_names:
                    for t, rn in zip(rel_types, rel_names):
                        norm_t = "पति" if t == "पतिका" else t
                        r_rels.append(f"{norm_t}: {rn}")
                    continue

            # Fallback single relation
            m_rel = REL_RE.search(line)
            if m_rel:
                rel_type = m_rel.group(1).strip()
                if rel_type == "पतिका":
                    rel_type = "पति"
                r_rels.append(f"{rel_type}: {m_rel.group(2).strip()}")

            # 4. Multi-house split across columns
            if any(k in line for k in ["मकान", "प्रकान", "भकान"]):
                splits = [re.sub(r"(?:फोटो\s*उपलब्ध|फोटो|उपलब्ध|[|.]).*", "", h).strip(" :：") for h in re.split(r"(?:मकान|प्रकान|भकान)\s*(?:संख्या|संयम|संकया|संया|नं|क्र)\s*[:：]?", line) if h.strip()]
                if splits:
                    for h in splits:
                        clean_h = h.strip()
                        if clean_h:
                            r_houses.append(clean_h)
                    continue

            # 5. Multi-age/gender split across columns
            m_ag = re.findall(r"([0-9०-९]{1,3})\s*[^0-9०-९\n]{1,15}?(महिला|पुरु[ष|थ|स]|अन्य)", line)
            if m_ag:
                for ag, gn in m_ag:
                    r_age_gens.append((ag.strip(), gn.strip()))

        n_voters = max(len(r_epics), len(r_names), len(r_serials), len(r_age_gens))
        for i in range(n_voters):
            rec_ser = r_serials[i] if i < len(r_serials) else ""
            rec_epic = r_epics[i] if i < len(r_epics) else ""
            rec_name = r_names[i] if i < len(r_names) else ""
            rec_rel = r_rels[i] if i < len(r_rels) else ""
            rec_house = r_houses[i] if i < len(r_houses) else ""
            rec_age = r_age_gens[i][0] if i < len(r_age_gens) else ""
            rec_gen = r_age_gens[i][1] if i < len(r_age_gens) else ""

            records.append(
                VoterRecord(
                    serial=rec_ser,
                    epic=rec_epic,
                    name=rec_name,
                    relation=rec_rel,
                    house=rec_house,
                    age=rec_age,
                    gender=rec_gen,
                    header_context=header_ctx,
                )
            )

    return header_ctx, records

