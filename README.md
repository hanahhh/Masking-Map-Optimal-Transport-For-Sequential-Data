# Masking Map Optimal Transport For Sequential Data (MOT)

MOT introduce an optimal transport model based on constraint masks to guide the matching of other points in OT. There are two types of masks: the linear mask aims to preserve the relative positions of elements, and the nonlinear mask models the value relationships of elements with their adjacent neighbors.

The illustration for the mask is shown below:
(https://github.com/hanahhh/Masking-Map-Optimal-Transport-For-Sequential-Data/main/Images/masking_map.png)

# Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup.py develop
```

# Dataset

LMSOT: python .\Experiments\LMSOT.py --algorithm_path .\Config\Algorithm\LMSOT.json --data_path .\Config\Data\DataMultivariate.json --result_path .\Results\Experiment\LMSOT\LMSOT.txt
NMSOT: python .\Experiments\NMSOT.py --algorithm_path .\Config\Algorithm\NMSOT.json --data_path .\Config\Data\DataUnivariate.json --result_path .\Results\Experiment\NMSOT\NMSOT.txt
LMOT: python .\Experiments\LMOT.py --algorithm_path .\Config\Algorithm\LMOT.json --data_path .\Config\Data\DataUnivariate.json --result_path .\Results\Experiment\LMOT\LMOT.txt
NMOT: python .\Experiments\NMOT.py --algorithm_path .\Config\Algorithm\NMOT.json --data_path .\Config\Data\DataMultivariate.json --result_path .\Results\Experiment\NMOT\NMOT.txt
