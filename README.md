# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/tree/main/paper)


## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `sklearn_tabular` | Sklearn Built-in Tabular Classification Datasets... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/sklearn_tabular/demo/data_code_demo.ipynb) |
| `openml_mixed_ty` | OpenML Mixed-Type Classification Benchmarks... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/openml_mixed_ty/demo/data_code_demo.ipynb) |
| `openml_classif` | OpenML Numeric-Only Classification Benchmarks... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/openml_classif/demo/data_code_demo.ipynb) |
| `pid_synergy_gra` | Pairwise PID Synergy Graph Construction Across 12 ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/pid_synergy_gra/demo/method_code_demo.ipynb) |
| `sg_figs_exp` | SG-FIGS vs Baselines: Synergy-Guided Oblique Tree ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/sg_figs_exp/demo/method_code_demo.ipynb) |
| `sg_figs_eval` | SG-FIGS: Fold-Aware Synergy-Guided Oblique Splits ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/sg_figs_eval/demo/method_code_demo.ipynb) |
| `sg_figs_eval` | SG-FIGS Comprehensive Statistical Evaluation... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/sg_figs_eval/demo/eval_code_demo.ipynb) |
| `split_quality` | Information-Theoretic Split Quality Audit... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/split_quality/demo/eval_code_demo.ipynb) |
| `synergygraph_ev` | Synergy Graph Sufficiency Analysis... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/synergygraph_ev/demo/eval_code_demo.ipynb) |
| `bootstrap_taxon` | Bootstrap Effect Sizes with Failure Taxonomy... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part/blob/main/bootstrap_taxon/demo/eval_code_demo.ipynb) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/AMGrobelnik/ai-invention-ac2586-synergy-guided-oblique-splits-using-part.git
cd ai-invention-ac2586-synergy-guided-oblique-splits-using-part

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
