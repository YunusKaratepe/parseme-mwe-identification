# French (FR) Model Comparison - Ablation Study

## Overall Performance

| Model | MWE F1 | TOK F1 | Improvement |
|-------|--------|--------|-------------|
| **Baseline** (BERT only) | 0.6437 | 0.7815 | - |
| **+ POS Injection** | 0.6566 | 0.7780 | +1.29% MWE F1 |
| **+ POS + CRF** | 0.6786 | 0.7935 | +3.49% MWE F1 |
| **+ POS + CRF + Focal Loss** | 0.6761 | 0.7914 | +3.24% MWE F1 |

---

## Continuity & Token Statistics

| Type | Baseline | + POS | + POS + CRF | + POS + CRF + Focal | Best Model |
|------|----------|-------|-------------|---------------------|------------|
| **Continuous** | 0.6916 | 0.6985 | 0.7180 | 0.7151 | + POS + CRF ✓ |
| **Discontinuous** | 0.4537 | 0.4977 | 0.5253 | 0.5291 | + POS + CRF + Focal ✓ |
| **Multi-token** | 0.6790 | 0.6880 | 0.7068 | 0.7048 | + POS + CRF ✓ |
| **Single-token** | 0.3000 | 0.3146 | 0.3415 | 0.3373 | + POS + CRF ✓ |

---

## Category-Level Performance

| Category | Baseline | + POS | + POS + CRF | + POS + CRF + Focal | Best Model |
|----------|----------|-------|-------------|---------------------|------------|
| **AdpID** | 0.8960 | 0.8480 | 0.6929 | 0.6713 | Baseline ✓ |
| **ConjID** | 0.9333 | 0.8000 | 0.0000 | 0.0000 | Baseline ✓ |
| **NID** | 0.6328 | 0.6424 | 0.6291 | 0.6213 | + POS ✓ |
| **AdvID** | 0.6320 | 0.6266 | 0.5597 | 0.5714 | Baseline ✓ |
| **PronID** | 0.6667 | 0.6667 | 0.0000 | 0.0000 | Baseline/+ POS ✓ |
| **IRV** | 0.5833 | 0.6800 | 0.3871 | 0.6000 | + POS ✓ |
| **NV.VID** | 0.5333 | 0.7059 | 0.0000 | 0.0000 | + POS ✓ |
| **LVC.full** | 0.4124 | 0.4086 | 0.3956 | 0.3571 | Baseline ✓ |
| **DetID** | 0.3333 | 0.4138 | 0.0000 | 0.0000 | + POS ✓ |
| **VID** | 0.3387 | 0.3793 | 0.2429 | 0.2319 | + POS ✓ |
| **AdjID** | 0.3200 | 0.3200 | 0.0000 | 0.0000 | Baseline/+ POS ✓ |
| **LVC.cause** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - |
| **NV.LVC.full** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - |

---

## Key Insights

### ✅ What Worked

1. **POS Injection** (+1.29% overall):
   - Strong improvements on discontinuous MWEs (+4.4%)
   - Better on IRV (+9.7%), NV.VID (+17.3%), DetID (+8.1%)
   - Modest gains on continuous MWEs (+0.7%)

2. **CRF Layer** (+2.2% over POS):
   - **Best overall performance**: 67.86% MWE F1
   - Excellent on discontinuous MWEs (+2.8% over POS)
   - Best continuous MWE score (71.80%)
   - Strong on structural patterns (multi-token +1.9%)

3. **Focal Loss** (slight regression vs CE):
   - Overall: 67.61% vs 67.86% (-0.25%)
   - **Best discontinuous MWE score**: 52.91% (+0.38% over CRF+CE)
   - Similar continuous performance: 71.51% vs 71.80%
   - Mixed category results

### ⚠️ Focal Loss Analysis

**Improvements over CRF+CE:**
- Discontinuous MWEs: +0.38% (0.5253 → 0.5291)
- IRV: +21.3% (0.3871 → 0.6000)
- AdvID: +1.2% (0.5597 → 0.5714)

**Regressions:**
- Overall F1: -0.25% (0.6786 → 0.6761)
- VID: -1.1% (0.2429 → 0.2319)
- LVC.full: -3.9% (0.3956 → 0.3571)
- AdpID: -2.2% (0.6929 → 0.6713)

**Conclusion**: Focal Loss helps with discontinuous MWEs (its intended purpose) but slightly hurts overall performance. The trade-off may be worthwhile if discontinuous MWE detection is critical.

### ⚠️ Concerning Pattern (CRF Models)

**Both CRF+CE and CRF+Focal show severe degradation on many categories:**
- ConjID: 93.3% → 0.0%
- PronID: 66.7% → 0.0%
- NV.VID: 70.6% → 0.0%
- DetID: 41.4% → 0.0%
- AdjID: 32.0% → 0.0%

**Possible causes:**
1. CRF over-constraining rare category transitions
2. Evaluation bug with category predictions in CRF mode
3. Different test data splits
4. Training convergence issues

**Despite category issues, overall F1 improved**, suggesting CRF helps with boundary detection even if category accuracy suffers.

---

## Recommended Configuration

### For Maximum Overall Performance
**POS + CRF (CE Loss)** - 67.86% F1
- Best continuous MWE performance (71.80%)
- Strong discontinuous MWE handling (52.53%)
- Highest token-level F1 (79.35%)

### For Maximum Discontinuous MWE Detection
**POS + CRF + Focal Loss** - 67.61% F1 overall, 52.91% discontinuous
- Best discontinuous MWE score (+0.38% over CE)
- Good balance on rare categories (IRV: 60.0%)
- Slight trade-off on overall performance (-0.25%)

### For Category-Specific Tasks
**POS only** or **Baseline**
- Better category distribution
- More reliable on rare categories
- ConjID/AdpID particularly strong
- No category zeroing issues

---

## Ablation Summary

| Component | Contribution |
|-----------|-------------|
| BERT Baseline | 64.37% F1 |
| + POS Injection | +1.29% F1 |
| + CRF Layer (CE) | +2.20% F1 |
| + Focal Loss (vs CE) | -0.25% F1 overall, +0.38% discontinuous |
| **Best Overall** | **67.86% F1 (POS + CRF + CE)** |
| **Best Discontinuous** | **52.91% (POS + CRF + Focal)** |

---

## Focal Loss vs Cross-Entropy (with POS + CRF)

| Metric | CE Loss | Focal Loss | Winner |
|--------|---------|------------|--------|
| Overall F1 | 0.6786 | 0.6761 | CE ✓ |
| Continuous | 0.7180 | 0.7151 | CE ✓ |
| Discontinuous | 0.5253 | 0.5291 | Focal ✓ |
| Multi-token | 0.7068 | 0.7048 | CE ✓ |
| Single-token | 0.3415 | 0.3373 | CE ✓ |

**Verdict**: Focal Loss achieves its design goal (better discontinuous MWE detection) but at the cost of overall performance. Use CE for general purpose, Focal if discontinuous MWEs are critical.
