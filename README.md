# The Self-Disclosure Dilemma: Context May Not Improve LLM Advice

Anonymous code and data accompanying the submitted paper.

| Paper material | Files |
| --- | --- |
| Theme definitions, inclusion and exclusion rules | [Qualitative codebook](qualitative/codebook.md) |
| Discovery cases D01–D12, full pairs and author memos | [Discovery](qualitative/discovery.md) |
| Verification cases V01–V12, full pairs, quotations and alternative readings | [Verification](qualitative/verification.md), [status matrix](qualitative/verification_matrix.csv) |
| Response-distance results, behavioral tables and reliability | [Saved results](data/), [methods and file guide](METHODS.md) |
| Questions, disclosures, judge rubric and generation settings | [Instruments](instruments/) |
| Reproduction and extension | [Code](code/) |

The qualitative [file guide](qualitative/README.md) explains the paper/workbook
theme identifiers and the difference between the paper's seven-case reduced-agency
grouping and the workbook's four-case rescuing code. Author annotations are
preserved as recorded.

## Reproduce

```sh
python3 code/verify.py
python3 -m pip install -r requirements.txt
python3 code/reproduce.py
python3 code/make_figures.py
```

The integrity check uses only Python's standard library. Numerical reproduction
and plotting run offline; outputs go to `outputs/`. They check all 33 distance
means and intervals, the three behavioral tables, six condition radars, word
counts and the paper's 643-response Opus–Qwen reliability analysis. The release
includes complete primary response texts and rephrasing pairs.

Optional fresh encoding and API collection are described in [METHODS.md](METHODS.md).
Encoder weights and provider credentials are not bundled. Fresh inference is
separate from reproducing the archived results.
