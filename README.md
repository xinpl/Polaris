# Polaris

## Overview

This repository hosts the code and data for NIPS'18 paper: Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections.

## Prerequisites

You can install all the required packages natively, but we recommend using [conda](https://conda.io/miniconda.html).

1. Create an environment:

   ```
   conda create -n polaris python=3.6 tensorflow keras pandas requests
   source activate polaris
   ```
2. Install `namedlist`, `jsonlines`, and `svgwrite`:
   ```
   pip install namedlist jsonlines svgwrite
   ```

3. Install [Gurobi](http://www.gurobi.com/). Academic licenses are free.
   
4. Install cleverhans:
   ```
   pip install cleverhans
   ```

5. Install [magenta](https://github.com/tensorflow/magenta). (Required by **drawing tutoring**).


## Running the experiments

* To run **mortgage underwriting**:
   ```
   python -m fanniemae.mortgage_exp ./fanniemae/data/imb_100k.test ./fanniemae/models/model_5_200 100 100
   ```
* To run **solver performance prediction**:
   ```
   python -m proof.proof_explain ./proof/models/8x100.h5 100
   ```
* To run **drawing tutoring**:
   ```
   python -m gold_cat.cat_exp ./gold_cat/model/dis/cat_model_mix-9000 ./gold_cat/model/gen/
   ```

## Training the models

Instead of using pre-trained models, you can train your own models. (Coming soon)
