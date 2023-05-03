# OT-4-change-detection

This repository contains all the necessary to reproduce the work "*Optimal Transport for Change Detection on LiDAR Point Clouds*" by Marco Fiorucci, Peter Naylo rand Makoto Yamada. 
<!-- You can find the paper [here](not working) (UNPUBLISHED). -->

# Usage

## Installation
Most of the required packages are listed in the `environment.yml` file.

## `txt` files

The pipeline accepts `txt` files as input in the following format: `NAME_0.txt` and `NAME_1.txt`.
Modify the variable `paired_txt` in the file `nextflow/main.nf` and run:

``` make data_txt ```

## `ply` files

The pipeline accepts `ply` files as input in the following format: `NAME_0.ply` and `NAME_1.ply`.
Modify the variable `paired_ply` in the file `nextflow/main.nf` and run:

``` make data_ply ```

## Output files

They can be found in the csv file `result/benchmark.csv`.

