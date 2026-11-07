"""Export a .pptx (PowerPoint) or .docx (Word) to PDF with Microsoft Office through COM (Windows only).

    python report/office_export.py deck.pptx [out.pdf]"""
import sys
from pathlib import Path
import pythoncom, win32com.client as w32


def pptx_to_pdf(src, dst):
    pythoncom.CoInitialize()
    app = w32.DispatchEx("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(str(src), WithWindow=False)
        pres.SaveAs(str(dst), 32)          # ppSaveAsPDF
        pres.Close()
    finally:
        app.Quit()


def docx_to_pdf(src, dst):
    pythoncom.CoInitialize()
    word = w32.DispatchEx("Word.Application"); word.Visible = False; word.DisplayAlerts = 0
    try:
        d = word.Documents.Open(str(src)); d.SaveAs2(str(dst), FileFormat=17); d.Close(False)
    finally:
        word.Quit()


if __name__ == "__main__":
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_suffix(".pdf")
    (pptx_to_pdf if src.suffix.lower() == ".pptx" else docx_to_pdf)(src, dst)
    print("saved", dst)
