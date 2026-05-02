# Day-Ahead Wind Power Ramp Events Prediction with Extreme-Value–Driven Learning and Confidence-Aware Detection
The repo is the official implementation for the paper: https://ieeexplore.ieee.org/abstract/document/11419886
# Introduction
Wind power ramp events (WPREs)—sudden, large-magnitude changes in wind generation—pose significant threats to power system stability. Accurate day-ahead prediction of WPREs is critical for proactive scheduling, reserve allocation, and grid security. However, existing methods often struggle with three key limitations: they underperform on rare and extreme events, lack sensitivity to multi-scale temporal dynamics, and provide little to no uncertainty quantification. To overcome these challenges, we propose a novel day-ahead WPREs prediction method that integrates three innovations. First, we design an Extreme-Value-Driven Learning model that decomposes wind power into multiple temporal frequency bands, enabling the simultaneous capture of long-term trends and short-term changes. Second, we introduce a tailored extreme-based loss function that rebalances model learning from average-case accuracy toward greater sensitivity to ramp events. Third, we develop a wind power–oriented conformal inference method that produces model-adaptive confidence intervals with formal statistical guarantees, enabling a confidence-aware ramp detection algorithm. Unlike conventional approaches that prioritize mean error metrics, our method is tailored to the operational significance of ramp events. Remarkably, experimental results on two real-world wind power datasets show that our method improves WPREs prediction accuracy by up to 29.6\% in terms of F1-score with negligible added computational cost.
# Code
1.Install Pytorch and necessary dependencies
```bash
pip install -r requirements.txt
```
2.The datasets can be obtained from SDWPF and GEFcom2014

3.Run dataload.py to generate Train, Val and Test sets in Temporal File

4.Run Run.py file to train and test the model
# EVDL
For the code of EVDL model you can see Hiformer for the details (https://github.com/valuestdio/Hiformer)
# Appendix 
Appendix can be found in folder: Appendix
