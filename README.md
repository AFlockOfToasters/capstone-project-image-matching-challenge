# Two Eyes See More - A Capstone Project by Dieter Janzen and Bernd Ackermann

Position-based image matching is used in 3D scanning of real objects under normally calibrated conditions. Using [LoFTR](https://zju3dv.github.io/loftr/), we match images regardless of size, lighting conditions, obstacles, and even photo filters, enabling the first step in digital 3D preservation of monuments and landmarks from mixed public images.

The [EDA notebook](notebooks/EDA.ipynb) gives an overview over the dataset.

The [LoFTR notebook](models/LoFTR/LoFTR.ipynb) demonstrates how to run LoFTR with PyTorch and how to plot matched images. The standalone [Python script](models/LoFTR/LoFTR.py) can be used to calculate matches for all possible image pairs in a given folder.

A dashboard, created using Plotly Dash, makes it easy to navigate through the dataset and plot matches for all image pairs. To run it, navigate to the [Viz_playground](models/Viz_playground/) and run:
```BASH
python app.py
```
The Dashboard can then be reached in a browser at [127.0.0.1:8050](http://127.0.0.1:8050).

## Requirements:

- pyenv with Python 3.9.8
- Data from the [Image Matching Challenge 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022/data).
### Setup

Use the requirements file in this repo to create a new environment:

```BASH
make setup

#or

pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
