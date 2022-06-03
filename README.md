# A Robust Neural Network for Extracting Dynamics from Time-Resolved Electrostatic Force Microscopy Data

## Accompanying code and example Jupyter notebook as prepared by Madeleine D. Breshears in collaboration with coauthors Rajiv Giridharagopal, Justin Pothoof, and David S. Ginger.

Department of Chemistry, University of Washington, Seattle, WA, USA, 98195

In order to extract dynamics from time-resolved electrostatic force microscopy (trEFM) data, we developed, trained, and tested an artificial neural network to intake arrays of data and output a time constant, $\\tau$ ($\mu s$), that describes the dynamics of interest. To read more about trEFM as a technique and more about this work in particular, see citations below.

## How to get started:
- Clone this repository or download the .ipynb and .py files directly. 
- Launch the Jupyter notebook and follow the included instructions on how to download the example data. (Please also note the documentation included in the NNFFtrEFM.py file.)

## Required Packages

`PyTorch`: `pip install torch`

`PyTorch Lightning`: `pip install pytorch-lightning`

`SHAP`: `pip install shap`


## Citations
1. Breshears, M.D., Giridharagopal, R., Pothoof, J., Ginger, D.S. A Robust Neural Network for Extracting Dynamics from Time-Resolved Electrostatic Force Microscopy Data. *Submitted.* **2022**.
2. Giridharagopal, R.; Precht, J. T.; Jariwala, S.; Collins, L.; Jesse, S.; Kalinin, S. v.; Ginger, D. S. Time-Resolved Electrical Scanning Probe Microscopy of Layered Perovskites Reveals Spatial Variations in Photoinduced Ionic and Electronic Carrier Motion. ACS Nano 2019, 13 (3), 2812–2821. https://doi.org/10.1021/acsnano.8b08390.
3. Giridharagopal, R.; Rayermann, G. E.; Shao, G.; Moore, D. T.; Reid, O. G.; Tillack, A. F.; Masiello, D. J.; Ginger, D. S. Submicrosecond Time Resolution Atomic Force Microscopy for Probing Nanoscale Dynamics. Nano Letters 2012, 12 (2), 893–898. https://doi.org/10.1021/nl203956q.
4. Karatay, D. U.; Harrison, J. S.; Glaz, M. S.; Giridharagopal, R.; Ginger, D. S. Fast Time-Resolved Electrostatic Force Microscopy: Achieving Sub-Cycle Time Resolution. Review of Scientific Instruments 2016, 87 (5), 053702. https://doi.org/10.1063/1.4948396.
