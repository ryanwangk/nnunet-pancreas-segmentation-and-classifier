# Pancreas Segmentation and Subtype Classifier
A multitask model for pancreas segmentation and subtype classification.
Built on nnUNetv2 with a 3D ResNet-M encoder


Input: Preprocessed 3D CT scans (`.nii.gz`)
> For more info on input format, please checkout the [nnUNetv2 repository](https://github.com/MIC-DKFZ/nnUNet)

Output: Segmentation mask (`.nii.gz`) and predicted subtype labels (`.csv`)

## Environments and Requirements

Project trained on:
- Google Colab
- GPU: A100
- Python: 3.12

## Installation

1. Install nnUNetv2 ([nnUNetV2 installation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md))
```python
pip install nnunetv2 dynamic-network-architectures
```

2. Replace the original files with the files in `src`

   In `nnUNetv2`, replace:
   - `nnUNetTrainer.py`
   - `data_loader.py`
   - `predict_from_raw_data.py`
   
   In `dynamic-network-architechtures`, replace:
   - `unet.py`
  
3. Run like you would with the base version of [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)


## Dataset

Prepare data based on [nnUNetv2 requirements](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)

Also include a `subtype_labels.csv` with true subtypes for each training case

Structure I used: 

```
Dataset310_Pancreas/
│── imagesTr/
│    ├── quiz_0_041_0000.nii.gz
│    ├── quiz_0_060_0000.nii.gz
│── labelsTr/
│    ├── quiz_0_041.nii.gz
│    ├── quiz_0_060.nii.gz
│── subtype_labels.csv
```

## Preprocessing

Run nnUNetv2 preprocessing command with `ResEncM`
```bash
nnUNetv2_plan_and_preprocess -d <dataset_id> -pl nnUNetPlannerResEncM --verify_dataset_integrity
```

## Training

1. To train the model run this command:

   ```bash
   nnUNetv2_train <dataset_id> 3d_fullres <fold> \
     -p nnUNetResEncUNetMPlans
   ```

   Note: The model uses `wandb` to track metrics, make sure you're logged in before training
   
   ```python
   import wandb
   wandb.login()
   ```
   
   You can download trained models here:
   
   - [Model 1](https://drive.google.com/file/d/1AjwlB7_Tr9rp2EFrqzK_HiIsnqT_W8XV/view?usp=share_link) 
   

2. [Colab Jupyter Notebook](https://colab.research.google.com/drive/1v-huIhSNIyNtM44VgRGEwqYQDi_FTW1r?usp=sharing)


## Inference

Run inference as you would with [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

Command I used:

```python
nnUNetv2_predict \
  -d <dataset_id> \
  -i <input_folder> \
  -o <output_folder> \
  -f <fold> \
  -c 3d_fullres \
  -chk <model>
  -p nnUNetResEncUNetMPlans
```

## Evaluation

Code to compute evaluation metrics could be found in the [Colab Notebook](https://colab.research.google.com/drive/1v-huIhSNIyNtM44VgRGEwqYQDi_FTW1r?usp=sharing)


## Results

Our method achieves the following performance on [Brain Tumor Segmentation (BraTS) Challenge](https://www.med.upenn.edu/cbica/brats2020/)

| Model name       |  Whole Pancreas DSC  | Pancreas Lesion DSC | Classification F1 |
| ---------------- | :----: | :--------------------: |:--------------------:|
| model425         | 0.86  |  0.5062                | 0.6018 |


## References and Acknowledgement
1. nnUNetv2: https://github.com/MIC-DKFZ/nnUNet/tree/master
2. MICCAI-Reproducibility-Checklist: https://github.com/JunMa11/MICCAI-Reproducibility-Checklist
