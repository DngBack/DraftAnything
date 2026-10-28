# Audit Kleister-Charity (5 field có ứng viên đóng)

## train (1729 tài liệu)

| field | null | absent | single | repeated | competing | #giá trị ứng viên (median) | oracle F1 | first | last | frequent | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| income | 1% | 20% | 0% | 0% | 79% | 228 | 89.0 | 0.5 | 0.2 | 2.5 | 1.7 |
| spending | 1% | 20% | 0% | 0% | 78% | 228 | 88.6 | 0.0 | 0.4 | 0.6 | 0.5 |
| date | 0% | 5% | 1% | 86% | 8% | 6 | 97.4 | 77.2 | 61.6 | 88.9 | – |
| number | 0% | 5% | 11% | 58% | 26% | 1 | 97.4 | 79.5 | 63.2 | 77.9 | – |
| postcode | 2% | 30% | 7% | 14% | 48% | 2 | 82.0 | 52.7 | 25.2 | 40.1 | – |

## dev-0 (440 tài liệu)

| field | null | absent | single | repeated | competing | #giá trị ứng viên (median) | oracle F1 | first | last | frequent | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| income | 1% | 21% | 0% | 0% | 78% | 252 | 88.1 | 0.5 | 0.2 | 1.6 | 0.2 |
| spending | 1% | 20% | 0% | 0% | 79% | 252 | 88.9 | 0.0 | 0.0 | 0.7 | 0.2 |
| date | 0% | 3% | 1% | 89% | 7% | 8 | 98.3 | 81.6 | 59.7 | 91.0 | – |
| number | 1% | 4% | 6% | 61% | 29% | 2 | 98.0 | 76.7 | 57.5 | 75.1 | – |
| postcode | 3% | 20% | 8% | 16% | 52% | 3 | 88.5 | 62.4 | 25.1 | 46.9 | – |
