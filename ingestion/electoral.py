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


def parse_electoral_records(text: str) -> tuple[str, list[VoterRecord]]:
    """Parse raw OCR text from a voter grid page into atomic VoterRecords.

    Uses row-based segmentation: an electoral page consists of stacked rows
    of 1 to 3 voter cards. Segmenting lines into local row blocks bounds
    any OCR-dropped field strictly to that row, completely preventing
    off-by-N field shifts from cascading down the page.

    Returns:
        (header_context, list_of_records)
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
        m_ser = SERIAL_RE.search(l)
        is_serial = bool(m_ser and int(m_ser.group(1)) < 2500 and len(m_ser.group(1)) <= 4)
        is_epic = bool(EPIC_RE.search(l))

        # A new row starts when we encounter a serial or EPIC *after* having already
        # collected voter details (name/house/age/etc.) in the current row.
        if (is_serial or is_epic) and cur_has_data:
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
            m_ser = SERIAL_RE.search(line)
            m_ep = EPIC_RE.search(line)
            m_rel = REL_RE.search(line)
            m_ag = AGE_GEN_RE.search(line)

            if m_ep:
                r_epics.append(m_ep.group(1).strip())
            elif m_ser and int(m_ser.group(1)) < 2500 and len(m_ser.group(1)) <= 4:
                r_serials.append(m_ser.group(1).strip())
            elif m_rel:
                rel_type = m_rel.group(1).strip()
                if rel_type == "पतिका":
                    rel_type = "पति"
                r_rels.append(f"{rel_type}: {m_rel.group(2).strip()}")
            elif "मकान" in line:
                matches = HOUSE_NO_PATTERN.findall(line)
                if matches:
                    for m in matches:
                        clean_m = m.strip()
                        if clean_m:
                            r_houses.append(clean_m)
                else:
                    h_val = re.sub(r"^.*?मकान\s*[^:\d०-९A-Za-z]*[:：]?\s*", "", line)
                    h_val = re.sub(r"(?:फोटो\s*उपलब्ध|फोटो|उपलब्ध|[|]).*$", "", h_val).strip()
                    if h_val:
                        r_houses.append(h_val)
            elif m_ag:
                r_age_gens.append((m_ag.group(1).strip(), m_ag.group(2).strip()))
            elif "नाम" in line and not any(k in line for k in ["विधानसभा", "भाग", "अनुभाग", "लोकसभा"]):
                v_name = re.sub(r"^.*?नाम\s*[:：]?\s*", "", line).strip()
                if v_name:
                    r_names.append(v_name)

        n_voters = max(len(r_epics), len(r_names))
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
