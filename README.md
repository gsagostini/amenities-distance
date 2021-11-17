# Accessibility in American Cities: Computing Network Distances Between Amenities and Census Tracts

In this repository I explore scalable and fast methods to compute distances between POIs (urban amenities) and centroids of census tracts in US cities. This work has been produced with Talia Kaufmann and Trivik Verma.

## Installation

To use the code in this repository, the creation of a new virtual environment is recommended. This can be done via the enviroment.yml file:
```
conda env create -f environment.yml
```
The environment `amenities-distance` will be created and can be activated with `conda activate urban-graphlets`. Installation via `pip` is not recommend due to constraints in the OSMNx library.

## Repository Setup

```
├── README.md                      <- README with project goals and installation guidelines.
│
├── environment.yml                <- Environment export generated with `conda env export > environment.yml`
│
├── .gitignore                     <- Avoids uploading certain files
│
├── data
│   ├── d1_raw               
│   ├── d2_processed
│   └── d3_results
│
├── notebooks                      <- Jupyter notebooks. Naming convention is date MMDD (for ordering) and a short description.
│
└── src                            <- Source code for use in this project, which can be imported as modules into the notebooks

```

## Data Sources

## References

