# DCDT

DCDT: A Domain-Generalized Conditional Diffusion Transformer for Cross-Subject EEG Emotion Recognition



# Datasets

The public available datasets (SEED and SEED-IV) can be downloaded from the https://bcmi.sjtu.edu.cn/home/seed/index.html

To facilitate data retrieval, the data from the first session of all subjects is utilized in both datasets, the file structure of the datasets should be like:

```
ExtractedFeatures/
    1/
eeg_feature_smooth/
    1/
```

Kindly change the file path in the main.py



## Usage of DCDT



Run , and The results will be recorded in TensorBoard. The argument for the is set to be for the SEED dataset, and for the SEED-IV dataset, respectively.`python main.py` `dataset_name` `seed3` `seed4` 
