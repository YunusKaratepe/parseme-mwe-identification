# MWE Type Analysis Report

**Generated**: December 9, 2025
**Dataset**: PARSEME 2.0 Subtask 1
**Languages Analyzed**: 16

## 📊 Overall Statistics

**Total MWEs Across All Languages**: 259,525

- **Continuous MWEs**: 258,059 (99.4%)
- **Discontinuous MWEs**: 1,466 (0.6%)

✅ **Discontinuous MWEs DO EXIST** in the data (0.6% of all MWEs)

## 📈 Dataset Distribution by Language

| Language | Train Sentences | Train Tokens | MWE Tokens | MWE Density | Dev Sentences                     | Dev Tokens | MWE Tokens | MWE Density | Test Sentences |
| -------- | --------------- | ------------ | ---------- | ----------- | --------------------------------- | ---------- | ---------- | ----------- | -------------- |
| **RO**   | 63,771          | 1,187,432    | 218,928    | 18.4%       | 7,085                             | 131,938    | 24,695     | 18.7%       | 364            |
| **KA**   | 25,956          | 202,773      | 6,061      | 3.0%        | 2,884                             | 22,530     | 684        | 3.0%        | 38,160         |
| **PL**   | 22,626          | 362,077      | 47,720     | 13.2%       | 2,513                             | 40,233     | 5,018      | 12.5%       | 1,127          |
| **HE**   | 17,553          | 293,048      | 48,970     | 16.7%       | 1,950 | 32,547     | 5,387      | 16.5%       | 1,188          |
| **LV**   | 10,705          | 144,974      | 9,737      | 6.7%        | 1,189                             | 16,097     | 1,067      | 6.6%        | 1,890          |
| **UK**   | 10,109          | 156,068      | 19,699     | 12.6%       | 1,123                             | 17,335     | 2,244      | 12.9%       | 846            |
| **SR**   | 8,901           | 102,992      | 32,198     | 31.3%       | 989                               | 11,440     | 3,822      | 33.4%       | 578            |
| **SL**   | 8,882           | 149,835      | 21,829     | 14.6%       | 986                               | 16,631     | 2,363      | 14.2%       | 1,156          |
| **SV**   | 4,303           | 70,092       | 9,348      | 13.3%       | 478                               | 7,784      | 1,016      | 13.1%       | 772            |
| **FR**   | 3,357           | 96,048       | 18,040     | 18.8%       | 373                               | 10,660     | 2,087      | 19.6%       | 354            |
| **FA**   | 1,576           | 44,095       | 15,384     | 34.9%       | 175                               | 4,987      | 1,780      | 35.7%       | 249            |
| **JA**   | 1,463           | 37,853       | 6,763      | 17.9%       | 162                               | 4,193      | 708        | 16.9%       | 338            |
| **EL**   | 1,380           | 34,819       | 2,671      | 7.7%        | 153                               | 3,863      | 300        | 7.8%        | 1,199          |
| **EGY**  | 431             | 5,597        | 468        | 8.4%        | 47                                | 610        | 64         | 10.5%       | 1,704          |
| **PT**   | 421             | 9,961        | 644        | 6.5%        | 46                                | 1,088      | 48         | 4.4%        | 1,219          |
| **GRC**  | -               | -            | -          | -           | -                                 | -          | -          | -           | 412            |
| **NL**   | 90              | 1,474        | 338        | 22.9%       | 10                                | 164        | 28         | 17.1%       | 400            |

**⚠️ Dataset Imbalance Concerns:**

- **RO dominance in training**: 63,771 sentences (37% of total) may cause language bias
- **KA test set anomaly**: 38,160 test sentences vs only 25,956 training sentences - severe distribution mismatch
- **Low-resource languages**: NL (90 train), PT (421 train), EGY (431 train) may underperform

## 🌍 Discontinuity by Language

| Language | Train MWEs | Discontinuous | Rate | Dev MWEs | Discontinuous | Rate |
| -------- | ---------- | ------------- | ---- | -------- | ------------- | ---- |
| **NL**   | 169        | 6             | 3.6% | 14       | 0             | 0.0% |
| **EGY**  | 234        | 4             | 1.7% | 32       | 0             | 0.0% |
| **UK**   | 9,855      | 164           | 1.7% | 1,122    | 18            | 1.6% |
| **SL**   | 10,923     | 147           | 1.3% | 1,183    | 18            | 1.5% |
| **FR**   | 9,044      | 66            | 0.7% | 1,046    | 11            | 1.1% |
| **FA**   | 7,693      | 44            | 0.6% | 890      | 4             | 0.4% |
| **RO**   | 113,330    | 597           | 0.5% | 12,764   | 74            | 0.6% |
| **EL**   | 1,337      | 7             | 0.5% | 150      | 0             | 0.0% |
| **LV**   | 4,872      | 24            | 0.5% | 534      | 3             | 0.6% |
| **SV**   | 4,676      | 20            | 0.4% | 508      | 1             | 0.2% |
| **HE**   | 24,571     | 93            | 0.4% | 2,701    | 8             | 0.3% |
| **PL**   | 23,876     | 80            | 0.3% | 2,514    | 11            | 0.4% |
| **PT**   | 322        | 1             | 0.3% | 24       | 0             | 0.0% |
| **SR**   | 16,119     | 47            | 0.3% | 1,912    | 6             | 0.3% |
| **JA**   | 3,382      | 9             | 0.3% | 354      | 0             | 0.0% |
| **KA**   | 3,032      | 3             | 0.1% | 342      | 0             | 0.0% |

## 📝 MWE Categories Distribution

| Category         | Count   | Percentage |
| ---------------- | ------- | ---------- |
| **MWE**          | 152,872 | 53.42%     |
| **AdpID**        | 24,503  | 8.56%      |
| **NID**          | 22,898  | 8.00%      |
| **AdvID**        | 17,989  | 6.29%      |
| **IRV**          | 14,006  | 4.89%      |
| **LVC.full**     | 11,057  | 3.86%      |
| **IAV**          | 9,530   | 3.33%      |
| **VID**          | 9,393   | 3.28%      |
| **ConjID**       | 8,952   | 3.13%      |
| **AdjID**        | 5,839   | 2.04%      |
| **DetID**        | 2,037   | 0.71%      |
| **PronID**       | 1,487   | 0.52%      |
| **LVC.cause**    | 1,447   | 0.51%      |
| **AV.IAV**       | 922     | 0.32%      |
| **IVPC.full**    | 722     | 0.25%      |
| **NV.LVC.full**  | 456     | 0.16%      |
| **NV.VID**       | 339     | 0.12%      |
| **IntjID**       | 328     | 0.11%      |
| **NV.IAV**       | 289     | 0.10%      |
| **IVPC.semi**    | 271     | 0.09%      |
| **MVC**          | 187     | 0.07%      |
| **AV.LVC.full**  | 110     | 0.04%      |
| **NV.IVPC.full** | 105     | 0.04%      |
| **AV.VID**       | 100     | 0.03%      |
| **NV.IVPC.semi** | 97      | 0.03%      |
| **NV.LVC.cause** | 95      | 0.03%      |
| **NV.MVC**       | 30      | 0.01%      |
| **AV.IVPC.semi** | 28      | 0.01%      |
| **AV.IVPC.full** | 28      | 0.01%      |
| **AV.LVC.cause** | 23      | 0.01%      |
| **AV.IRV**       | 14      | 0.00%      |
| **NV.IRV**       | 10      | 0.00%      |

## 📏 MWE Length Distribution

| MWE Length (tokens) | Count   | Percentage |
| ------------------- | ------- | ---------- |
| 1                   | 7,059   | 5.30%      |
| 2                   | 103,710 | 77.81%     |
| 3                   | 19,282  | 14.47%     |
| 4                   | 2,622   | 1.97%      |
| 5                   | 427     | 0.32%      |
| 6                   | 158     | 0.12%      |
| 7                   | 18      | 0.01%      |
| 8                   | 10      | 0.01%      |
| 9                   | 3       | 0.00%      |
| 10                  | 1       | 0.00%      |
| 12                  | 1       | 0.00%      |
| 13                  | 1       | 0.00%      |

## 🔍 Discontinuous MWE Examples

**PL (train)** - Type: `MWE`

- Tokens: obronie ... staję
- Positions: [10, 12]
- Gaps: [1] token(s) between components

**PL (train)** - Type: `MWE`

- Tokens: na ... cierpień
- Positions: [2, 5]
- Gaps: [2] token(s) between components

**PL (dev)** - Type: `MWE`

- Tokens: się ... echem
- Positions: [25, 27]
- Gaps: [1] token(s) between components

**PL (dev)** - Type: `MWE`

- Tokens: ślady ... pójdzie
- Positions: [1, 3]
- Gaps: [1] token(s) between components

**FR (train)** - Type: `MWE`

- Tokens: circulation ... les ... travailleurs
- Positions: [10, 12, 13]
- Gaps: [1] token(s) between components

**FR (train)** - Type: `MWE`

- Tokens: s ... agisse
- Positions: [35, 37]
- Gaps: [1] token(s) between components

**FR (dev)** - Type: `MWE`

- Tokens: posons ... question
- Positions: [6, 8]
- Gaps: [1] token(s) between components

**FR (dev)** - Type: `MWE`

- Tokens: les ... minutes
- Positions: [10, 14]
- Gaps: [3] token(s) between components

**EL (train)** - Type: `MWE`

- Tokens: σε ... επαφή
- Positions: [5, 8]
- Gaps: [2] token(s) between components

**EL (train)** - Type: `MWE`

- Tokens: αμαρτιών ... δίνει
- Positions: [2, 27]
- Gaps: [24] token(s) between components

## 📋 Detailed Language Statistics

### PL

**Training Set**:

- Sentences: 22,626
- Tokens: 371,875
- MWE tokens: 25,016 (6.73% density)
- Continuous MWEs: 23,796
- Discontinuous MWEs: 80 (0.3%)

**Top MWE Categories**:

- MWE: 13,285
- IRV: 3,522
- AdvID: 2,461
- LVC.full: 2,381
- AdpID: 1,002

**Dev Set**:

- Sentences: 2,513
- MWE tokens: 2,626
- Discontinuous MWEs: 11 (0.4%)

### FR

**Training Set**:

- Sentences: 3,357
- Tokens: 78,306
- MWE tokens: 10,655 (13.61% density)
- Continuous MWEs: 8,978
- Discontinuous MWEs: 66 (0.7%)

**Top MWE Categories**:

- MWE: 6,476
- NID: 1,452
- AdvID: 964
- AdpID: 577
- LVC.full: 433

**Dev Set**:

- Sentences: 373
- MWE tokens: 1,240
- Discontinuous MWEs: 11 (1.1%)

### EL

**Training Set**:

- Sentences: 1,380
- Tokens: 34,739
- MWE tokens: 1,531 (4.41% density)
- Continuous MWEs: 1,330
- Discontinuous MWEs: 7 (0.5%)

**Top MWE Categories**:

- MWE: 893
- LVC.full: 202
- VID: 141
- ConjID: 101
- AdvID: 90

**Dev Set**:

- Sentences: 153
- MWE tokens: 161
- Discontinuous MWEs: 0 (0.0%)

### PT

**Training Set**:

- Sentences: 421
- Tokens: 8,367
- MWE tokens: 384 (4.59% density)
- Continuous MWEs: 321
- Discontinuous MWEs: 1 (0.3%)

**Top MWE Categories**:

- MWE: 220
- AdvID: 33
- LVC.full: 27
- NID: 24
- ConjID: 23

**Dev Set**:

- Sentences: 46
- MWE tokens: 27
- Discontinuous MWEs: 0 (0.0%)

### RO

**Training Set**:

- Sentences: 63,771
- Tokens: 1,273,320
- MWE tokens: 115,698 (9.09% density)
- Continuous MWEs: 112,733
- Discontinuous MWEs: 597 (0.5%)

**Top MWE Categories**:

- MWE: 68,716
- AdpID: 16,509
- AdvID: 8,346
- IAV: 6,949
- NID: 6,815

**Dev Set**:

- Sentences: 7,085
- MWE tokens: 12,959
- Discontinuous MWEs: 74 (0.6%)

### SL

**Training Set**:

- Sentences: 8,882
- Tokens: 178,550
- MWE tokens: 12,036 (6.74% density)
- Continuous MWEs: 10,776
- Discontinuous MWEs: 147 (1.3%)

**Top MWE Categories**:

- MWE: 6,632
- IRV: 1,248
- AdpID: 894
- AdvID: 863
- IAV: 561

**Dev Set**:

- Sentences: 986
- MWE tokens: 1,292
- Discontinuous MWEs: 18 (1.5%)

### SR

**Training Set**:

- Sentences: 8,901
- Tokens: 180,963
- MWE tokens: 16,733 (9.25% density)
- Continuous MWEs: 16,072
- Discontinuous MWEs: 47 (0.3%)

**Top MWE Categories**:

- MWE: 9,006
- NID: 2,710
- IRV: 1,660
- LVC.full: 799
- AdvID: 768

**Dev Set**:

- Sentences: 989
- MWE tokens: 1,984
- Discontinuous MWEs: 6 (0.3%)

### SV

**Training Set**:

- Sentences: 4,303
- Tokens: 67,997
- MWE tokens: 5,082 (7.47% density)
- Continuous MWEs: 4,656
- Discontinuous MWEs: 20 (0.4%)

**Top MWE Categories**:

- MWE: 2,304
- AdvID: 619
- IVPC.full: 387
- LVC.full: 327
- VID: 301

**Dev Set**:

- Sentences: 478
- MWE tokens: 554
- Discontinuous MWEs: 1 (0.2%)

### UK

**Training Set**:

- Sentences: 10,109
- Tokens: 164,932
- MWE tokens: 10,809 (6.55% density)
- Continuous MWEs: 9,691
- Discontinuous MWEs: 164 (1.7%)

**Top MWE Categories**:

- MWE: 6,008
- IAV: 1,017
- LVC.full: 769
- AdpID: 717
- AdvID: 687

**Dev Set**:

- Sentences: 1,123
- MWE tokens: 1,242
- Discontinuous MWEs: 18 (1.6%)

### NL

**Training Set**:

- Sentences: 90
- Tokens: 1,358
- MWE tokens: 177 (13.03% density)
- Continuous MWEs: 163
- Discontinuous MWEs: 6 (3.6%)

**Top MWE Categories**:

- MWE: 99
- IAV: 19
- AdvID: 18
- VID: 14
- IVPC.full: 14

**Dev Set**:

- Sentences: 10
- MWE tokens: 16
- Discontinuous MWEs: 0 (0.0%)

### EGY

**Training Set**:

- Sentences: 431
- Tokens: 4,487
- MWE tokens: 246 (5.48% density)
- Continuous MWEs: 230
- Discontinuous MWEs: 4 (1.7%)

**Top MWE Categories**:

- MWE: 129
- AdpID: 34
- VID: 31
- AdvID: 14
- NV.VID: 9

**Dev Set**:

- Sentences: 47
- MWE tokens: 33
- Discontinuous MWEs: 0 (0.0%)

### KA

**Training Set**:

- Sentences: 25,956
- Tokens: 505,857
- MWE tokens: 3,151 (0.62% density)
- Continuous MWEs: 3,029
- Discontinuous MWEs: 3 (0.1%)

**Top MWE Categories**:

- MWE: 1,640
- VID: 1,051
- NID: 298
- LVC.full: 97
- AdvID: 32

**Dev Set**:

- Sentences: 2,884
- MWE tokens: 353
- Discontinuous MWEs: 0 (0.0%)

### JA

**Training Set**:

- Sentences: 1,463
- Tokens: 27,832
- MWE tokens: 3,785 (13.60% density)
- Continuous MWEs: 3,373
- Discontinuous MWEs: 9 (0.3%)

**Top MWE Categories**:

- MWE: 1,177
- LVC.full: 1,055
- AdpID: 501
- NID: 371
- AdvID: 215

**Dev Set**:

- Sentences: 162
- MWE tokens: 398
- Discontinuous MWEs: 0 (0.0%)

### HE

**Training Set**:

- Sentences: 17,553
- Tokens: 330,649
- MWE tokens: 25,573 (7.73% density)
- Continuous MWEs: 24,478
- Discontinuous MWEs: 93 (0.4%)

**Top MWE Categories**:

- MWE: 14,171
- NID: 6,564
- ConjID: 1,325
- VID: 1,290
- LVC.full: 947

**Dev Set**:

- Sentences: 1,950
- MWE tokens: 2,843
- Discontinuous MWEs: 8 (0.3%)

### LV

**Training Set**:

- Sentences: 10,705
- Tokens: 175,331
- MWE tokens: 5,166 (2.95% density)
- Continuous MWEs: 4,848
- Discontinuous MWEs: 24 (0.5%)

**Top MWE Categories**:

- MWE: 2,837
- AdvID: 699
- ConjID: 462
- VID: 361
- PronID: 233

**Dev Set**:

- Sentences: 1,189
- MWE tokens: 566
- Discontinuous MWEs: 3 (0.6%)

### FA

**Training Set**:

- Sentences: 1,576
- Tokens: 35,544
- MWE tokens: 8,011 (22.54% density)
- Continuous MWEs: 7,649
- Discontinuous MWEs: 44 (0.6%)

**Top MWE Categories**:

- MWE: 3,857
- LVC.full: 1,660
- AdpID: 701
- NID: 648
- AdvID: 330

**Dev Set**:

- Sentences: 175
- MWE tokens: 931
- Discontinuous MWEs: 4 (0.4%)

## 💡 Key Insights

1. **Discontinuity is relatively rare** (0.6% overall)
2. **Most MWEs are short**: 103,710 are 2-word MWEs
3. **Category distribution varies by language**
4. ℹ️ **Discontinuous MWEs are rare** - current BIO tagging may be sufficient

---

**Conclusion**: Current BIO tagging approach is likely adequate for most cases
