
# CrystalNeuroimaging
## Interpretation of 3D CNNs for Brain MRI Data Classification
In this repository we provide *Jupyter Notebooks* to make classification of 3DCNN for task of gender patterns recognition using HCP T1-weighted MRI data. We also provide different methods of interpretation to research and undestand of gender-related brain differencies and some swap tests to check them. 

## DATASET
 Dataset - Human Connection Project (HCP).
 We worked on the full-sized T1 MPI images preprocessed in `Freesurfer` according to the HCP pipeline.
 Data contain 517 subjects, including 210 men and 307 women.
 
## Install

Here is the list of some libraries to execute the code:

 - python = 3.6
 - pytorch = 0.4
 - nilearn
 - numpy
 - scipy 
 - matplotlib
 - scikit-image
 - jupyter

You can install them via `conda` (`anaconda`), e.g.
```python
conda install jupyter
```


## GradCAM

Interpretation with Grad CAM:
  - Mean over Female
  
![](image/mean0.png)

  - Mean over Male
  
![](image/mean1.png)

## Guided backpropagation

Interpretation with Guided backpropogation:
  - Mean over Female
  
![](image/gbmean0.png)

  - Mean over Male
  
![](image/gbmean1.png)

