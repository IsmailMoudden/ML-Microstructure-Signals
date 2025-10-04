# Research Paper - ML Microstructure Signals

This directory contains the research paper and all associated materials for the ML Microstructure Signals project.

## 📁 Directory Structure

```
paper/
├── README.md                    # This file
├── sources/                     # LaTeX source files
│   ├── main.tex                # Main LaTeX document
│   └── references.bib          # Bibliography
├── figures/                     # All figures and visualizations
│   ├── feature_correlation.png
│   ├── feature_importance.png
│   ├── model_performance.png
│   ├── backtest_results.png
│   └── order_book.png
└── ../ML_Microstructure_Signals_Research_Paper.pdf  # Final PDF (at project root)
```

## 📄 Paper Details

- **Title**: ML Microstructure Signals: Predicting Short-Term Mid-Price Moves from Order Book Features
- **Author**: Ismail Moudden
- **Email**: ismail.moudden1@gmail.com
- **Version**: 1.0
- **Date**: October 2024

## 🔧 Compilation

To compile the LaTeX document:

```bash
cd sources/
tectonic main.tex
```

The compiled PDF will be generated as `main.pdf` in the sources directory.

## 📊 Figures

All figures are stored in the `figures/` directory:

- **feature_correlation.png**: Feature correlation matrix heatmap
- **feature_importance.png**: Feature importance ranking from LightGBM
- **model_performance.png**: Model performance comparison charts
- **backtest_results.png**: Equity curve and drawdown analysis
- **order_book.png**: Limit order book structure visualization

## 📚 References

The bibliography is maintained in `sources/references.bib` and includes:
- Academic papers on microstructure analysis
- Machine learning references
- High-frequency trading literature
- Technical implementation papers

## 🎯 Paper Sections

1. **Introduction** - Problem statement and contributions
2. **Data and Features** - Data sources and feature engineering
3. **Models and Methodology** - ML models and labeling strategy
4. **Backtesting Framework** - Signal generation and execution
5. **Results and Analysis** - Performance evaluation and visualizations
6. **Robustness Analysis** - Walk-forward and sensitivity analysis
7. **Conclusion** - Summary and future work

## 📖 Access

The final PDF is available at the project root as:
**`ML_Microstructure_Signals_Research_Paper.pdf`**

This makes it easily accessible without navigating through directories.
