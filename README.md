# CrystalNeuroimaging
## Interpretation of 3D CNNs for Brain MRI Data Classification
Using HCP T1-weighted MRI data we tried classification of 3DCNN for task of gender patterns recognition. And then we applied different methods of interpretation to research and  undestand of gender-related brain differencies. 

## DATASET
 Dataset - Human Connection Project (HCP).
 We worked on the full-sized T1 MPI images preprocessed in `Freesurfer` according to the HCP pipeline.
 Data contain 517 subjects, including 210 men and 307 women.
 
 ## CNN Model

To train the 3D CNN models use 

```bash
3DCNN.ipynb
```

![](image/3DCNN.png)

[CometMl:](https://www.comet.ml/polina/mri-interpretation/view/uw5eiUdqrH5ArXAKBHGA1FKIr)

Train and validate loss

![TrainLoss](image/train_lossVSstep.jpeg) ![ValLoss](image/validate_lossVSstep.jpeg)

Train and validate accuracy

![TrainAcc](image/train_accVSstep.jpeg) ![ValAcc](image/validate_accVSstep.jpeg)


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

## Model Randomization Test 
### GradCam

![](image/ModelRandTestGradCam.png)

### Guided backpropagation

![](image/ModelRandTestGB.png)

## Date Randomization Test 


![](image/DataRandomTest.png)
