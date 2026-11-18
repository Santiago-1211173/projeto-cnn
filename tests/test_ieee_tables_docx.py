"""
Test and validation suite for outputs/figures_ieee/ieee_tables_section4b.docx.
Verifies IEEE TNNLS formatting, XML borders, typography, column counts, merges, and scientific values.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt
from docx.oxml.ns import qn

DOCX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures_ieee", "ieee_tables_section4b.docx")

def test_docx_structure():
    assert os.path.exists(DOCX_PATH), f"File {DOCX_PATH} does not exist"
    doc = docx.Document(DOCX_PATH)
    
    # Check Section Margins
    sec = doc.sections[0]
    assert abs(sec.page_width.inches - 8.5) < 0.01
    assert abs(sec.left_margin.inches - 0.67) < 0.01
    assert abs(sec.right_margin.inches - 0.67) < 0.01
    usable_width = sec.page_width.inches - sec.left_margin.inches - sec.right_margin.inches
    assert abs(usable_width - 7.16) < 0.02, f"Usable width is {usable_width}, expected 7.16 in"

    # Check Captions
    paras = [p for p in doc.paragraphs if p.text.strip()]
    assert len(paras) == 4, f"Expected 4 caption paragraphs, got {len(paras)}"
    assert paras[0].text.strip() == "TABLE I"
    assert "PREQUENTIAL ACCURACY" in paras[1].text
    assert paras[2].text.strip() == "TABLE II"
    assert "CONTINUAL LEARNING RESILIENCE" in paras[3].text

    # Check Caption Formatting
    for p in paras:
        assert p.alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            assert r.font.name == "Times New Roman"
            assert r.font.size == Pt(8)

    assert abs(paras[1].paragraph_format.space_after.pt - 4.0) < 0.1
    assert abs(paras[3].paragraph_format.space_after.pt - 4.0) < 0.1

    # Check Tables
    assert len(doc.tables) == 2, f"Expected 2 tables, got {len(doc.tables)}"
    t1, t2 = doc.tables[0], doc.tables[1]

    # TABLE I: 16 rows, 10 cols
    assert len(t1.rows) == 16, f"Table I rows: {len(t1.rows)}, expected 16"
    assert len(t1.columns) == 10, f"Table I cols: {len(t1.columns)}, expected 10"

    # TABLE II: 16 rows, 7 cols
    assert len(t2.rows) == 16, f"Table II rows: {len(t2.rows)}, expected 16"
    assert len(t2.columns) == 7, f"Table II cols: {len(t2.columns)}, expected 7"

    # Verify Table Widths & Borders
    for t_idx, table in enumerate([t1, t2]):
        # Check Header
        header_cells = table.rows[0].cells
        for cell in header_cells:
            tcBorders = cell._tc.get_or_add_tcPr().find(qn("w:tcBorders"))
            assert tcBorders is not None
            top_b = tcBorders.find(qn("w:top"))
            bot_b = tcBorders.find(qn("w:bottom"))
            left_b = tcBorders.find(qn("w:left"))
            right_b = tcBorders.find(qn("w:right"))
            # Top rule: sz="12" (1.5 pt)
            assert top_b is not None and top_b.get(qn("w:sz")) == "12"
            # Mid rule: sz="6" (0.75 pt)
            assert bot_b is not None and bot_b.get(qn("w:sz")) == "6"
            # No vertical borders
            assert left_b is not None and left_b.get(qn("w:val")) == "none"
            assert right_b is not None and right_b.get(qn("w:val")) == "none"

        # Check Group Separators (Rows 6 and 11)
        for r_idx in [6, 11]:
            for cell in table.rows[r_idx].cells:
                tcBorders = cell._tc.get_or_add_tcPr().find(qn("w:tcBorders"))
                top_b = tcBorders.find(qn("w:top"))
                assert top_b is not None and top_b.get(qn("w:sz")) == "4", f"Row {r_idx} top border sz is {top_b.get(qn('w:sz'))}"

        # Check Bottom Rule (Row 15)
        for cell in table.rows[15].cells:
            tcBorders = cell._tc.get_or_add_tcPr().find(qn("w:tcBorders"))
            bot_b = tcBorders.find(qn("w:bottom"))
            assert bot_b is not None and bot_b.get(qn("w:sz")) == "12", f"Row 15 bottom border sz is {bot_b.get(qn('w:sz'))}"

    # Check Content of Table I
    # Row 5 (MNIST B4)
    r5 = t1.rows[5]
    assert "B4: Proposed Active RL" in r5.cells[1].text
    assert r5.cells[2].text == "95.70"
    assert r5.cells[3].text == "90.70"
    assert r5.cells[4].text == "55.70"
    assert r5.cells[5].text == "38.90"
    assert r5.cells[6].text == "27.30"
    assert r5.cells[7].text == "61.66"
    assert r5.cells[8].text == "61.48"
    assert "2.80 × 10" in r5.cells[9].text and "-3" in r5.cells[9].text

    # Row 10 (CIFAR-10 B4)
    r10 = t1.rows[10]
    assert "7.00 × 10" in r10.cells[9].text and "-4" in r10.cells[9].text

    # Row 15 (CIFAR-100 B4)
    r15 = t1.rows[15]
    assert "2.25 × 10" in r15.cells[9].text and "-8" in r15.cells[9].text

    # Check Content of Table II
    # B4 Peak RAM indicator
    r5_t2 = t2.rows[5]
    assert r5_t2.cells[6].text == "9.9241 (O(1))"
    r10_t2 = t2.rows[10]
    assert r10_t2.cells[6].text == "9.9330 (O(1))"
    r15_t2 = t2.rows[15]
    assert r15_t2.cells[6].text == "8.8165 (O(1))"

    # Verify B4 Bold Styling
    for table in [t1, t2]:
        for b4_row_idx in [5, 10, 15]:
            for col_idx in range(1, len(table.columns)):
                cell = table.rows[b4_row_idx].cells[col_idx]
                runs = cell.paragraphs[0].runs
                assert len(runs) > 0
                for r in runs:
                    assert r.font.bold is True, f"Cell ({b4_row_idx}, {col_idx}) run '{r.text}' is not bold!"

    print("[PASS] All IEEE TNNLS formatting, booktabs borders, typography, and scientific data checks passed!")

if __name__ == "__main__":
    test_docx_structure()

