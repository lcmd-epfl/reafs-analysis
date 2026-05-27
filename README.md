# ReaFS Analysis

This repository contains the analysis code to reproduce the results and plots from [the main manuscript and supplementary information](doi.org).

## Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [reafs](https://github.com/lcmd-epfl/reafs)
- ipykernel
- ipywidgets

See `pyproject.toml` for complete dependency specifications. Most of the dependencies are installed with `reafs`.

### Installation

#### Using `uv` (recommended)

```bash
uv sync
```

#### Notes

The [reafs](github.com) package is expected to be located in the parent directory. Modify `pyproject.toml` accordingly.


## Main Scripts

### Results & Plots

- **`plot_cc_model_cards_1.py`** - Generate parts of the model "cards" for the featurization comparison as in the main manuscript.
- **`plot_cc_model_cards_SI_1.py`** - Supplementary information version of model cards for the featurization comparison
- **`plot_data_diagno_1.py`** - Diagnostic plots for data analysis
- **`plot_data_g23_outliers_1.py`** - Outlier analysis and visualization
- **`plot_fs_common_1.py`** - Visualize common feature selections across analyses
- **`plot_fs_sequential_extension_1.py`** - Sequential feature selection extension plots
- **`plot_oa_prospect_1.py`** - Prospective analysis "cards" parts.
- **`plot_oa_prospect_SI_1.py`** - Supplementary information for the prospective analysis

### Utilities

- **`svg_composer_feat.py`** - Compose the final plot from the featurization comparison from all the parts
- **`svg_composer_prosp.py`** - Compose the final plot from the prospective analysis from all the parts

## Running the Analysis

To run a specific analysis script:

```bash
uv run python script_name.py
```

Or,

```bash
python script_name.py
```

## Output

Results and generated figures are saved to the `results/` directory.



