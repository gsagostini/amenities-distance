# Accessibility in American Cities: Computing Network Distances Between Amenities and Census Tracts

In this repository I explore scalable and fast methods to compute distances between POIs (urban amenities) and centroids of census tracts in US cities. This work has been produced with Talia Kaufmann and Trivik Verma.

## Installation

To use the code in this repository, the creation of a new virtual environment is recommended. This can be done via the enviroment.yml file:
```
conda env create -f environment.yml
```
The environment `amenities-distance` will be created and can be activated with `conda activate amenities-distance`. Installation via `pip` is not recommend due to constraints in the OSMNx library.

## Repository Setup

```
├── README.md                      <- README with project goals and installation guidelines.
│
├── environment.yml                <- Environment export generated with `conda env export > environment.yml`
│
├── data                           <- Data used and produced. Not all files are uploaded, check data section.
│   ├── d1_raw            
│   ├── d2_processed-safegraph
|   ├── d3_intermediary
│   └── d4_final-OD-matrices
│
├── notebooks                      <- Jupyter notebooks for exploratory data analysis. Naming convention is date MMDD (for ordering) and a short description.
│
└── src                            <- Source code for use in this project, which can be imported as modules into the notebooks. Includes also bash scripts for job submission in a cluster.
    └── processing-scripts         <- Outdated scripts that perform smaller tasks (prepare matrices, trim rows, etc)

```

## Data Sources

Data 

## References

