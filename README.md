# Network-informed Spatial Weights (NSW)

This repository provides a Python implementation of **Network-informed Spatial Weights (NSW)** — a methodology that integrates graph-based overlapping community detection into spatial econometric modeling. Instead of relying solely on geographic proximity, NSW constructs spatial weight matrices from functional linkages (e.g., commuting flows, trade networks) and includes tools for testing and correcting potential endogeneity in the weights.

## Overview

Spatial econometric models often depend on spatial weight matrices to represent interdependencies between units. Traditional schemes, such as contiguity or distance thresholds, may fail to capture non-geographic but economically important linkages. NSW addresses this limitation by:

- Representing spatial entities as nodes in a directed, weighted economic interaction network.
- Detecting overlapping communities using network science algorithms.
- Assigning primary communities based on strongest internal linkages.
- Constructing community-based spatial weight matrices.
- Testing for weight matrix endogeneity using Rao's score test.
- Applying corrections through control functions or PCA proxies when necessary.

This approach allows researchers to incorporate richer theoretical foundations into their models and better capture real-world interdependencies.

## Features

- **Flexible community detection**: Leiden multiplex, OSLOM, weighted cliques, ego-splitting.
- **Overlapping membership support**: Capture complex regional roles in multiple functional groups.
- **Primary community assignment**: Based on largest internal flow strength.
- **Community-based spatial weights**: Row-standardized for econometric use.
- **Endogeneity diagnostics**: Rao's score test for spatial weight matrices.
- **Endogeneity correction**: Control function and PCA proxy approaches.
- **Complete pipeline**: From raw flow data to model estimation.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/USERNAME/network-informed-spatial-weights.git
cd network-informed-spatial-weights
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- networkx
- pandas
- numpy
- libpysal
- spreg
- scikit-learn
- matplotlib (optional, for visualizations)

## Function Documentation

### `run_sem_with_community_weights`
Runs the full pipeline for constructing network-informed spatial weights, estimating a spatial error model, and performing optional endogeneity testing and correction.

**Parameters:**
- `nodes` (`pd.DataFrame`): Table of spatial units and associated attributes.
- `flows` (`pd.DataFrame`): Table of directed flows with origin, destination, and weight.
- `id_col` (`str`): Column in `nodes` containing unique IDs for spatial units.
- `target_col` (`str`): Name of dependent variable column in `nodes`.
- `feature_cols` (`Sequence[str]`): List of independent variable column names in `nodes`.
- `origin_col` (`str`): Column in `flows` with origin node IDs.
- `dest_col` (`str`): Column in `flows` with destination node IDs.
- `weight_col` (`str`): Column in `flows` with flow weights.
- `community_method` (`str`): Overlapping community detection method. Options:
  - `"leiden_multiplex"` (default, recommended)
  - `"oslom"`
  - `"weighted_cliques"`
  - `"ego_splitting"`
- `community_kwargs` (`dict`): Extra parameters for community detection.
- `sem_kwargs` (`dict`): Extra parameters for spatial econometric model estimation.
- `Z` (`np.ndarray`, optional): Variables driving W formation (for formal endogeneity test).
- `X2` (`np.ndarray`, optional): Exogenous predictors for Z.
- `n_pca_components` (`int`): PCA components to use for correction if endogenous and no Z provided (default 1).
- `test_endogeneity` (`bool`): Whether to perform endogeneity test (default True).

**Returns:**
`dict` containing:
- `'base_model'`: Initial spatial error model.
- `'corrected_model'`: Endogeneity-corrected model (if applied).
- `'endogeneity_test'`: Test results dictionary.
- `'primary'`: Primary community assignments.
- `'W'`: Spatial weights matrix.
- `'G'`: Flow graph.
- `'overlaps'`: Overlapping community memberships.


## Usage Example

```python
from NSW import run_sem_with_community_weights

results = run_sem_with_community_weights(
    nodes=nodes_df,
    flows=flows_df,
    target_col="gdp",
    feature_cols=["broadband_access", "high_tech_employment"],
    community_method="leiden_multiplex",
    sem_kwargs={"spat_diag": True}
)

# Access results
base_model = results['base_model']
corrected_model = results['corrected_model']
endogeneity_test = results['endogeneity_test']

print(f"Endogeneity detected: {endogeneity_test['is_endogenous']}")
print(f"P-value: {endogeneity_test['p_value']}")
```

## Example Applications

- Regional economic performance modeling with commuting or trade flows.
- Innovation diffusion analysis based on R&D collaboration networks.
- Environmental spillover studies using shared resource or pollution networks.
- Transport accessibility studies with origin–destination flow data.


## Data Availability

The methodology can be applied to any node–flow dataset where edges represent meaningful functional linkages between spatial units. Due to licensing restrictions, the dataset used in the associated publication cannot be shared. Example code is provided for demonstration purposes with placeholder or synthetic data.


## Citation

If you use this code in your research, please cite:

> CITATION WHEN PUBLISHED

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

