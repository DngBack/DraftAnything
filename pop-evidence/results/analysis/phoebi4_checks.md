# phoebi4 pre-registered checks

Check 1: nearest phoebi6 pure culture to each phoebi4 pure culture (cosine of mean tile embeddings):

- b -> nearest bt (0.9208); expected bs; all: bs 0.883, bt 0.921, fj 0.919, ka 0.496, mx 0.755, pf 0.878
- f -> nearest mx (0.9035); expected fj; all: bs 0.850, bt 0.813, fj 0.762, ka 0.818, mx 0.903, pf 0.843
- k -> nearest ka (0.8699); expected ka; all: bs 0.701, bt 0.645, fj 0.616, ka 0.870, mx 0.760, pf 0.700
- p -> nearest fj (0.9506); expected pf; all: bs 0.920, bt 0.947, fj 0.951, ka 0.581, mx 0.810, pf 0.919

Check 2: max cosine of 2,000 phoebi4 images to 40,000 phoebi6 images: max 0.97599, 99th pct 0.96967; images above 0.999: 0

**Checks passed: False**