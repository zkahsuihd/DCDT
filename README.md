# DCDT

DCDT: A Domain-Generalized Conditional Diffusion Transformer for Cross-Subject EEG Emotion Recognition



# Update

- [2025.08.2] Release main codes
- [2025.08.1] This repo is created.



# Overview

![](C:\Users\ck\Desktop\images\frame.png)



# Installation

1. Clone Repo

   ```
   git clone https://github.com/zkahsuihd/DCDT
   cd DCDT
   ```

2. Create Conda Environment and Install Dependencies

   ```
   # create new conda env
   conda create -n dcdt python=3.9 -y
   conda activate dcdt
   
   # install python dependencies
   pip install -r requirements.txt
   ```





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



## result

| Method  | SEED  | SEED  | SEED-IV | SEED-IV |
| ------- | ----- | ----- | ------- | ------- |
|         | Avg.  | Std.  | Avg.    | Std.    |
| DG-ML   | 79.41 | 10.47 |         |         |
| MGFKD   | 87.51 | 7.68  | 68.79   | 8.25    |
| CCP     | 80.74 | 6.05  | 62.65   | 9.79    |
| LDLC    | 82.05 | 5.91  |         |         |
| WGAN-DA | 87.07 | 7.14  |         |         |
| CLISA   | 86.4  | 6.4   |         |         |
| DMMR    | 88.27 | 5.62  | 72.70   | 8.01    |
| DG-DANN | 84.30 | 8.32  |         |         |
| PPDA    | 86.70 | 7.10  |         |         |
| DCDT    | 89.07 | 5.40  | 72.95   | 7.89    |

Experimental results demonstrate that the DCDT framework achieves a comprehensive performance breakthrough in cross-subject EEG emotion recognition tasks. On the SEED dataset, DCDT attains an average accuracy of 89.07%±5.40, signiffcantly surpassing existing methods—exceeding the conventional domain adversarial model DG-DANN (84.30%±8.32) by 4.77%, and outperforming the state-of-the-art hybrid reconstruction approach DMMR (88.27%±5.62) by 0.80%. These results validate the effectiveness of the conditional diffusion feature decoupling mechanism in mitigating subject-speciffc noise interference, as multi-source domain conditional noise prediction enables the puriffcation of emotional representations and lays a solid foundation for cross-domain generalization.



![](C:\Users\ck\Desktop\images\model_accuracy_comparison.png)

