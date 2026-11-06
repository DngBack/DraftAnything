# Paper

`main.tex` uses the ACL template in review mode. Every number comes from `numbers.tex`, `tables/` and `fig/`, generated from `results/`:

```bash
.venv/bin/python -W ignore scripts/paper_tables.py   # needs data/decide_set.json, data/token_set.json and the results files
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

A key missing from `numbers.tex` stops the build. Do not type numbers into the prose.

**Status:** draft on an incomplete sample. Protocol D stopped at 218, 260 and 127 of 500 cells; Protocol T stopped at 6 to 13 of 200 cells and is not analysed. To finish, run `scripts/run_decide.py` and `scripts/run_token.py` again (they resume), rerun the commands above, and revise the prose against the new verdicts, since some sentences describe the current sample.

`verified_refs.md` records how much of each reference was read. Recheck numbers against the PDFs before quoting them.
