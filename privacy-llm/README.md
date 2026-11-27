# privacy-llm

Nghiên cứu **contextual privacy** của LLM agent. Bước đầu: phân tích bộ dữ liệu
**AirGapAgent-R** ([`parameterlab/leaky_thoughts`](https://huggingface.co/datasets/parameterlab/leaky_thoughts),
paper *Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers*, arXiv:2506.15674).

## Cấu trúc

```
privacy-llm/
├── privacy_llm/            # thư viện (không I/O ngoài data/, không side-effect khi import)
│   ├── config.py           # repo id, revision pin, tên file các split, đường dẫn
│   ├── download.py         # tải file + ghi MANIFEST.json (sha256)
│   ├── data.py             # load + validate schema -> DataFrame; nhóm field (taxonomy riêng)
│   ├── eda.py              # các hàm phân tích thuần (DataFrame in -> DataFrame/dict out)
│   └── plots.py            # các hình matplotlib (data in -> Figure out)
├── scripts/
│   ├── download_data.py    # python scripts/download_data.py [--force]
│   └── run_eda.py          # python scripts/run_eda.py  -> reports/eda/
├── tests/test_data.py      # invariants của dataset mà phân tích dựa vào
├── data/raw/               # file JSON gốc (git-ignored, chỉ commit MANIFEST.json)
└── reports/
    ├── EDA_REPORT.md       # báo cáo EDA (đọc file này trước)
    └── eda/                # output sinh tự động: figures/, tables/, summary.json
```

## Chạy lại

```bash
cd privacy-llm
pip install -r requirements.txt
python scripts/download_data.py                 # ~10 MB, revision f1d8f27
python -m unittest discover -s tests -v         # 9 tests
python scripts/run_eda.py                       # ghi reports/eda/
```
