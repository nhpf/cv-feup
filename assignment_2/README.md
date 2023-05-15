# Environment configuration

Setup a virtual environment and install the project dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Learn more about PyTorch configuration [here](https://pytorch.org/get-started/locally/).

## Context
This project uses YOLOv8 and DeepSORT to perform "Multiple Vessel Detection and Tracking in Harsh Maritime Environments".
It aims to recreate the results of a paper released this year by researches at INESC TEC.
You can learn more about the project and our achievements by reading the final report as described below.

## File structure

### `docs/`
 - Base paper, project proposal, references and final report
 - You should compile the `report.tex` with XeLaTeX

### `examples/`
 - Small scripts that work as minimal working examples of each relevant process present in the project
 - Visualize the labeled dataset, perform detection with YOLOv8, try different methods of data augmentation, etc.

### `tasks/`
 - Each script in this directory contains the isolated implementation of the major steps performed in the final Jupyter notebook
 - The order of execution is given by the number preceding each script, i.e. the first script is `1_process_singapore.py`

### `main.py`
 - Entry point for the project before the notebook is ready
 - It has useful comments that will aid us during the project development
