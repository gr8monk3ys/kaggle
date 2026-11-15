# US Federal Register Documents 2024-2026

**Kaggle:** [lorenzoscaturchio/us-federal-register-documents-2024-2026](https://www.kaggle.com/datasets/lorenzoscaturchio/us-federal-register-documents-2024-2026)

## Description

Every document published in the US Federal Register from 1 January 2024 to
31 October 2026, pulled from the official Federal Register API on 2026-11-15.
This is real data, not generated: 76,223 documents, one row each. They are
8,161 final rules, 4,765 proposed rules, 62,291 notices and 1,006
presidential documents.

The window covers the last year of the Biden administration and the first 21
months of the second Trump administration. The data records the January 2025
regulatory freeze, the jump in executive orders, the drop in final rules, and
the October 2025 shutdown.

## Files

| File | Rows | Contents |
|------|------|----------|
| `federal_register_documents.csv` | 76,223 | Title, abstract, action, agencies, significance, pages, effective and comment dates, CFR titles and topics, dockets, RINs, executive order fields, page views |

## Columns

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `document_number` | string | 0.0% | 76,223 | 2023-26792, 2023-27783, 2023-27901 |
| `publication_date` | date | 0.0% | 706 | 2024-01-02, 2024-01-03, 2024-01-04 |
| `type` | string | 0.0% | 4 | Notice, Rule, Proposed Rule |
| `subtype` | string | 98.7% | 7 | Proclamation, Determination, Memorandum |
| `title` | string | 0.0% | 44,300 | Notice of Solicitation of Ap, Carrier Automated Tariffs, Notice of Funding Opportunit |
| `abstract` | string | 28.0% | 40,501 | The Rural Business-Cooperati, The Federal Maritime Commiss, The Rural Utilities Service |
| `action` | string | 30.6% | 4,550 | Notice., Final rule., Notice of Funding Opportunit |
| `agency` | string | 0.0% | 147 | Agriculture Department, Federal Maritime Commission, Small Business Administratio |
| `agencies` | string | 0.0% | 433 | Agriculture Department/Rural, Federal Maritime Commission, Agriculture Department/Rural |
| `n_agencies` | integer | 0.0% | 10 | 2, 1, 3 |
| `significant` | boolean | 90.9% | 2 | True, False |
| `page_length` | integer | 0.0% | 237 | 6, 9, 5 |
| `effective_on` | date | 87.7% | 1,015 | 2024-02-01, 2024-03-01, 2024-01-01 |
| `days_to_effective` | integer | 87.7% | 493 | 30, 59, -1 |
| `comments_close_on` | date | 69.2% | 935 | 2023-03-03, 2024-01-25, 2024-02-01 |
| `comment_period_days` | integer | 69.2% | 191 | -305, 23, 30 |
| `topics` | string | 87.4% | 2,251 | Freight/Intermodal transport, Campaign funds, Administrative practice and |
| `n_topics` | integer | 0.0% | 55 | 0, 4, 1 |
| `cfr_titles` | string | 83.0% | 102 | 46, 11, 40 |
| `n_cfr_references` | integer | 0.0% | 37 | 0, 1, 30 |
| `n_dockets` | integer | 0.0% | 38 | 1, 0, 2 |
| `regulation_id_numbers` | string | 88.9% | 3,419 | 3072-AC86, 0648-BL53, 1545-BQ90 |
| `president` | string | 0.0% | 2 | Joseph R. Biden Jr., Donald Trump |
| `executive_order_number` | integer | 99.6% | 324 | 14115, 14116, 14117 |
| `signing_date` | date | 98.7% | 405 | 2023-12-27, 2023-12-28, 2023-12-29 |
| `is_correction` | boolean | 0.0% | 2 | False, True |
| `page_views` | integer | 0.0% | 4,569 | 563, 2648, 166 |

Pipe-separated list columns are shown with `/` in the samples above.
`significant` is blank when it is not reported. That covers every notice and
about half of rules, so blank means unknown, not false.

## What the starter notebook finds

`explore.ipynb` computes these from the full CSV:

- February 2025 published 48% fewer documents than the median month (2,308),
  after the 20 January regulatory freeze. October 2025 published 75% fewer,
  during the government shutdown.
- Per month in office, final rules fell from 275 (Biden) to 219 (Trump),
  significant final rules from 30 to 16, and Federal Register pages from 11,170
  to 7,507. Executive orders rose from 2.4 to 13.8 a month. The window holds
  325 executive orders, numbered 14115 to 14438.
- 6.7% of final-rule titles under Trump announce a rescission, repeal,
  removal, withdrawal or delay, against 1.8% under Biden. This is a keyword
  count, not a legal classification.
- Comment periods on proposed rules did not change. The median is 45 days
  under both administrations.
- A TF-IDF + logistic regression baseline trained on 2024-2025 predicts the
  `significant` label for 2026 rules with ROC-AUC 0.961 and PR-AUC 0.821. The
  base rate is 0.175.

## Caveats

- `page_views` is a count at snapshot time. Older documents have had longer to
  collect views.
- The Biden span in this window is about 13 months and the Trump span about
  21, so compare rates, not totals.
- Much of the volume is routine output, such as Coast Guard safety zones, FAA
  airworthiness directives and SEC self-regulatory filings. Filter by `agency`
  or `type` before reading a count as policy.
- `days_to_effective` and `comment_period_days` can be negative, for example
  when a correction or reopening refers to an earlier deadline (1,013 and 73
  rows).
- `abstract` is blank for 28% of documents, mostly notices and presidential
  documents.

## Reproducing

`python fetch_dataset.py` runs one API query per month, needs no key, and
takes about three minutes. It fails if any month returns fewer documents than
the API reports. Use `--start` and `--end` to change the window. Published
documents do not change, but `page_views` and `significant` can, so a re-run
gives slightly different values in those two columns.

## License and source

Federal Register API (https://www.federalregister.gov/developers), from the
Office of the Federal Register (NARA) and the Government Publishing Office.
Federal Register content is a US government work and is not under copyright
in the United States. This compilation is released under CC0 1.0. It is not an
official legal edition of the Federal Register.

## Suggested Use Cases

Text classification (agency, document type or significance from the
abstract), regulatory time series and forecasting, comparing administrations,
topic modelling of executive orders, and comment-period and effective-date
analysis.

## Tags

government, politics, law, nlp, united states
