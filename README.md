# Retail Promotional Event Analysis

## Measuring Marketing Effectiveness Through Statistical Testing

**Author:** Jacob Aguillard  
**Contact:** aguillarj@gmail.com  
**LinkedIn:** [linkedin.com/in/jacobaguillard](www.linkedin.com/in/jacob-aguillard-24851015a)

---

## Project Overview

This project analyzes the effectiveness of various promotional tactics on Return on Investment (ROI) across different retail departments. Using statistical methods including t-tests, chi-square tests, and effect size calculations, I identify which marketing interventions significantly impact sales performance—and which ones don't.

### Business Questions Answered

1. **Which promotional tactics significantly improve ROI?**
2. **Do certain tactics work better in specific departments?**
3. **How can we quantify the incremental value of each tactic?**
4. **Does day of week affect promotional performance?**

---

## Key Findings

| Promotional Tactic | Departments with Positive Effect | Key Insight |
|-------------------|----------------------------------|-------------|
| End Cap Placement | Food division departments | ~15-25% ROI improvement in snacks, frozen, beverages |
| Coupons | Snacks, Frozen, Beverages, Dairy | ~10-30% ROI improvement |
| Digital Signage | Mixed results | Effect varies significantly by department |
| QR Codes | Limited data | Insufficient adoption for conclusive results |

**Counterintuitive Finding:** Some tactics that appear beneficial overall actually hurt ROI in certain departments, highlighting the importance of department-specific strategies.

---

## Methods Used

### Statistical Analysis
- **Independent t-tests** - Compare means between groups with/without promotional tactic
- **Cohen's d effect size** - Quantify practical significance beyond p-values
- **Chi-square tests** - Test categorical associations (day of week effects)
- **Z-tests for proportions** - Compare conversion rates across segments

### Data Processing
- **Winsorization** - Handle outliers by capping extreme values at 5th/95th percentiles
- **Missing data analysis** - Assess data quality before analysis
- **IQR outlier detection** - Identify anomalous records

---

## Technical Stack

| Category | Tools |
|----------|-------|
| Languages | Python |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy (stats, chi2_contingency) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Project Structure

```
retail-promotion-analysis/
│
├── README.md                          # This file
├── retail_promotion_analysis.ipynb    # Main analysis notebook
├── generate_synthetic_data.py         # Synthetic data generator
├── synthetic_retail_promotions.csv    # Generated dataset (100K records)
└── requirements.txt                   # Python dependencies
```

---

## Dataset

This analysis uses **synthetic data** that preserves the statistical properties of real retail promotional event data while protecting proprietary information.

### Dataset Specifications
- **Records:** 100,000 promotional events
- **Departments:** 15 retail categories
- **Regions:** 6 geographic areas
- **Date Range:** 2 years of event data
- **Metrics:** Sales, costs, ROI, conversion rates, promotional flags

### Key Fields

| Field | Description |
|-------|-------------|
| `department` | Retail category (Snacks, Frozen, Electronics, etc.) |
| `roi_winsorized` | Return on investment, outliers capped |
| `sales_lift` | Incremental sales vs baseline |
| `conversion_rate` | Units sold / customer exposure |
| `end_cap` | Binary: premium shelf placement |
| `has_coupon` | Binary: coupon distributed |
| `digital_signage` | Binary: digital display used |

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scipy matplotlib seaborn jupyter
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/retail-promotion-analysis.git
cd retail-promotion-analysis

# Generate synthetic data (optional - CSV included)
python generate_synthetic_data.py

# Launch Jupyter and open the notebook
jupyter notebook retail_promotion_analysis.ipynb
```

---

## Sample Output

### T-Test Results: End Cap Effect on ROI

```
================================================================================
T-TEST ANALYSIS: END_CAP EFFECT ON ROI_WINSORIZED
================================================================================

Records analyzed: 100,000
Groups analyzed: 15

Significance levels: *** p<0.001, ** p<0.01, * p<0.05

SUMMARY:
  Statistically significant results (p<0.05): 8
  Groups where end_cap HELPS: 6
  Groups where end_cap HURTS: 2

GROUPS WHERE END_CAP SIGNIFICANTLY HELPS:
  Snacks & Candy: +0.342 (p=0.0001) ***
  Frozen Foods: +0.287 (p=0.0003) ***
  Beverages: +0.251 (p=0.0012) **
```

---

## Skills Demonstrated

- ✅ **Statistical Analysis** - Hypothesis testing, effect sizes, significance interpretation
- ✅ **Python Programming** - Clean, documented, reusable functions
- ✅ **Data Wrangling** - Handling messy data, outliers, missing values
- ✅ **Business Acumen** - Translating stats into actionable recommendations
- ✅ **Data Visualization** - Clear, informative charts
- ✅ **Documentation** - Well-structured code and explanations

---

## About Me

I'm a data analyst with 4+ years of experience in marketing analytics, specializing in:
- Statistical analysis and A/B testing
- Power BI dashboard development
- Python/SQL data engineering
- Translating data into business insights

**Currently seeking:** Senior Data Analyst, Analytics Manager, or Marketing Analytics roles

📧 aguillarj@gmail.com

---

## License

This project is available for educational and portfolio purposes. The synthetic data generator and analysis code are free to use and modify.
