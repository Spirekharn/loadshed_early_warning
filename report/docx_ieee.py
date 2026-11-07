"""Tiny helper to write an IEEE-conference-style (A4, two-column) Word document with python-docx,
then finish it in Microsoft Word (native equations, PDF export) through COM automation.

Layout follows the IEEE conference template (A4): Times New Roman 10 pt body, 24 pt title, margins
top 19 mm / bottom 43 mm / left-right 14.32 mm, two columns 88.6 mm wide with a 4.2 mm gap."""
import re
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

COL_W = Mm(88.57)
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")


def _set_cols(section, n, space_mm=4.22):
    sp = section._sectPr
    cols = sp.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols"); sp.append(cols)
    cols.set(qn("w:num"), str(n)); cols.set(qn("w:space"), str(int(space_mm / 25.4 * 1440)))


def _margins(section):
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.top_margin, section.bottom_margin = Mm(19), Mm(43)
    section.left_margin = section.right_margin = Mm(14.32)
    section.header_distance = section.footer_distance = Mm(12.7)


def rich(par, text, size=None, bold=False, italic=False, small_caps=False):
    """add text with a mini markup: **bold**, *italic*, _{subscript}, ^{superscript}"""
    for tok in TOKEN.split(text):
        if not tok:
            continue
        b, i, sub, sup, t = bold, italic, False, False, tok
        if tok.startswith("**") and tok.endswith("**"):
            b, t = True, tok[2:-2]
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            i, t = not italic, tok[1:-1]
        elif tok.startswith("_{"):
            sub, t = True, tok[2:-1]
        elif tok.startswith("^{"):
            sup, t = True, tok[2:-1]
        r = par.add_run(t)
        r.bold, r.italic = b, i
        r.font.subscript, r.font.superscript = sub, sup
        if small_caps:
            r.font.small_caps = True
        if size:
            r.font.size = Pt(size)
    return par


class IEEEDoc:
    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(10)
        st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Times New Roman")
        pf = st.paragraph_format; pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = 1.0
        _margins(self.d.sections[0]); _set_cols(self.d.sections[0], 1)
        self.h1 = 0; self.h2 = 0; self.fig = 0; self.tab = 0; self.eq = 0

    # ---------------------------------------------------------------- front matter
    def title(self, text):
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(text); r.font.size = Pt(24)

    def authors(self, people):
        """people: list of lists of lines; first line = name (11 pt), rest 10 pt (2nd/3rd italic)."""
        t = self.d.add_table(rows=1, cols=len(people)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, lines in zip(t.rows[0].cells, people):
            cell.paragraphs[0].text = ""
            for k, line in enumerate(lines):
                p = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(line); r.font.size = Pt(11 if k == 0 else 10); r.italic = k in (1, 2)
        p = self.d.add_paragraph(); p.paragraph_format.space_after = Pt(6)

    def start_columns(self):
        s = self.d.add_section(WD_SECTION.CONTINUOUS); _margins(s); _set_cols(s, 2)

    def end_columns(self):
        """a final continuous section break makes Word balance the two columns on the last page"""
        s = self.d.add_section(WD_SECTION.CONTINUOUS); _margins(s); _set_cols(s, 2)

    def abstract(self, text, label="Abstract"):
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Mm(3.5); p.paragraph_format.space_after = Pt(6)
        r = p.add_run(f"{label}—"); r.bold = True; r.italic = True; r.font.size = Pt(9)
        rich(p, text, size=9, bold=True)

    # ---------------------------------------------------------------- body
    def heading(self, text):
        self.h2 = 0
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format; pf.space_before = Pt(8); pf.space_after = Pt(4); pf.keep_with_next = True
        r = p.add_run(f"{ROMAN[self.h1]}. "); r.font.size = Pt(10)
        r = p.add_run(text); r.font.small_caps = True; r.font.size = Pt(10)
        self.h1 += 1

    def unnumbered_heading(self, text):
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format; pf.space_before = Pt(8); pf.space_after = Pt(4); pf.keep_with_next = True
        r = p.add_run(text); r.font.small_caps = True

    def subheading(self, text):
        p = self.d.add_paragraph()
        pf = p.paragraph_format; pf.space_before = Pt(6); pf.space_after = Pt(3); pf.keep_with_next = True
        r = p.add_run(f"{chr(65 + self.h2)}. {text}"); r.italic = True
        self.h2 += 1

    def para(self, text, indent=True, size=None, space_after=0):
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            p.paragraph_format.first_line_indent = Mm(3.5)
        p.paragraph_format.space_after = Pt(space_after)
        rich(p, text, size=size)
        return p

    def bullets(self, items, numbered=False):
        for k, it in enumerate(items, 1):
            p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format; pf.left_indent = Mm(5); pf.first_line_indent = Mm(-3.5)
            pf.tab_stops.add_tab_stop(Mm(5))
            p.add_run(f"{k})\t" if numbered else "•\t")
            rich(p, it)

    def equation(self, linear):
        """display equation; `linear` is Word's linear (UnicodeMath) format, built up later in Word."""
        self.eq += 1
        p = self.d.add_paragraph()
        pf = p.paragraph_format; pf.space_before = Pt(3); pf.space_after = Pt(3)
        pf.tab_stops.add_tab_stop(COL_W // 2, WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(COL_W - Mm(0.5), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\tEQ{" + linear + "}EQ\t" + f"({self.eq})")
        return self.eq

    def figure(self, path, caption, width_mm=86):
        self.fig += 1
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format; pf.space_before = Pt(6); pf.keep_with_next = True
        p.add_run().add_picture(str(path), width=Mm(width_mm))
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.space_after = Pt(8); c.paragraph_format.space_before = Pt(2)
        r = c.add_run(f"Fig. {self.fig}. "); r.font.size = Pt(8)
        rich(c, caption, size=8)
        return self.fig

    def table(self, title, header, rows, widths_mm, bold_rows=(), align=None, note=None):
        """IEEE table: 'TABLE I' + title (small caps, above), 8 pt body, rules above/below header and at the bottom."""
        self.tab += 1
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format; pf.space_before = Pt(6); pf.keep_with_next = True
        r = p.add_run(f"Table {ROMAN[self.tab - 1]}"); r.font.size = Pt(8); r.font.small_caps = True
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format; pf.space_after = Pt(3); pf.keep_with_next = True
        r = p.add_run(title); r.font.size = Pt(8); r.font.small_caps = True
        t = self.d.add_table(rows=1 + len(rows), cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "bottom"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000")
            borders.append(e)
        tblPr.append(borders)
        cm = OxmlElement("w:tblCellMar")
        for side, v in (("left", 50), ("right", 50), ("top", 10), ("bottom", 10)):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); cm.append(e)
        tblPr.append(cm)
        align = align or (["l"] + ["c"] * (len(header) - 1))
        for i, row in enumerate([header] + rows):
            tr = t.rows[i]._tr
            trPr = tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
            for j, val in enumerate(row):
                cell = t.cell(i, j); cell.width = Mm(widths_mm[j])
                par = cell.paragraphs[0]
                par.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                                 "r": WD_ALIGN_PARAGRAPH.RIGHT}[align[j]]
                par.paragraph_format.keep_with_next = i < len(rows)
                rich(par, str(val), size=8, bold=(i == 0 or i - 1 in bold_rows))
                if i == 0:
                    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
                    e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); b.append(e)
                    tcPr.append(b)
        if note:
            p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            rich(p, note, size=7)
        p = self.d.add_paragraph(); p.paragraph_format.space_after = Pt(4)
        return self.tab

    def references(self, entries):
        for k, e in enumerate(entries, 1):
            p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf = p.paragraph_format; pf.left_indent = Mm(6.5); pf.first_line_indent = Mm(-6.5)
            pf.tab_stops.add_tab_stop(Mm(6.5)); pf.space_after = Pt(1)
            r = p.add_run(f"[{k}]\t"); r.font.size = Pt(8)
            rich(p, e, size=8)

    def save(self, path):
        self.d.save(str(path))


def finish_in_word(docx_path, pdf_path):
    """Open in Word: turn every EQ{...}EQ marker into a native equation, save .docx again and export PDF."""
    import pythoncom, win32com.client as w32
    pythoncom.CoInitialize()
    word = w32.DispatchEx("Word.Application"); word.Visible = False; word.DisplayAlerts = 0
    try:
        D = word.Documents.Open(str(docx_path))
        u16 = lambda s: len(s.encode("utf-16-le")) // 2
        n = 0
        for p in list(D.Paragraphs):
            t = p.Range.Text
            if "EQ{" not in t:
                continue
            a, b = t.index("EQ{"), t.index("}EQ")
            s = p.Range.Start + u16(t[:a]); e = p.Range.Start + u16(t[:b + 3])
            r = D.Range(s, e); r.Text = t[a + 3:b]
            om = D.OMaths.Add(r); om.OMaths(1).BuildUp(); n += 1
        D.Repaginate()
        pages = D.ComputeStatistics(2)
        D.SaveAs2(str(docx_path), FileFormat=16)
        D.SaveAs2(str(pdf_path), FileFormat=17)
        D.Close(False)
    finally:
        word.Quit()
    return n, pages
