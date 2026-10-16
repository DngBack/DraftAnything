"""Fallback for poppler tools: python pdftool.py unite out.pdf in1.pdf in2.pdf | pages f.pdf | text f.pdf PAGE"""
import sys

from pypdf import PdfReader, PdfWriter

cmd, args = sys.argv[1], sys.argv[2:]
if cmd == "unite":
    w = PdfWriter()
    for f in args[1:]:
        w.append(f)
    w.write(args[0])
elif cmd == "pages":
    print(len(PdfReader(args[0]).pages))
elif cmd == "text":
    print(PdfReader(args[0]).pages[int(args[1]) - 1].extract_text())
