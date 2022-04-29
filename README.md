# PSOD (Point Saliency)
# Weakly-Supervised Salient Object Detection Using Point Supervison

![VIZ](https://user-images.githubusercontent.com/34783695/165979177-3ecef77f-d553-4747-97f7-2f7788c2adad.png)


![image](https://user-images.githubusercontent.com/34783695/159275127-1a6bd023-5b97-427a-9f5c-4b4854656415.png)


# Prerequisites
- Python 3.6
- Pytorch 1.7
- TensorboardX
- OpenCV 3.4
- Numpy 1.19

# Clone repository
```
git clone git@github.com:shuyonggao/PSOD.git
cd PSOD
```


# Download P-DUTS Dataset

* point supervised DUTS: [google](https://drive.google.com/file/d/1ZV2Bk1nZ3GRqcVvrabybSKT8N-1XsSH8/view?usp=sharing)

# Download Saliency map & Trained model
* saliency maps: [google](https://drive.google.com/file/d/1TqIOXidkxkhq9nI0KBMApREam-EMnnr-/view?usp=sharing)

* trained model: [google](https://drive.google.com/file/d/1S8za3FiPalP0wRqazjj060wm1Sc3XwrB/view?usp=sharing)

* initial pseudo-labels: [google](https://drive.google.com/file/d/1TqIOXidkxkhq9nI0KBMApREam-EMnnr-/view?usp=sharing)

* edge maps: [google](https://drive.google.com/file/d/1Juzi-TZJfrB9iv_4UOYs60qn2VpZ033h/view?usp=sharing)



# Train Model

1. [Point supervison](https://drive.google.com/file/d/1ZV2Bk1nZ3GRqcVvrabybSKT8N-1XsSH8/view?usp=sharing) and [edge maps]() are employed to generate [initial pseudo-labels](https://drive.google.com/file/d/1TqIOXidkxkhq9nI0KBMApREam-EMnnr-/view?usp=sharing) (we provided the initial pseudo-labels and you can use it directly for training).

    Run ```python utils/EdgePoint2gt.py```

2. Set the path of the [training images](http://saliencydetection.net/duts/), edge maps and initial pseudo-labels in ```dataset_1st.py```, ```dataset_2nd.py```, ```test_DUTS.py```

    Run ```bash train_nss_retrain.sh```
# Test Model

1. Set the path of the test data in ```test.py```.

2. Creat "out_2nd" folder and put the [trained model](https://drive.google.com/file/d/1S8za3FiPalP0wRqazjj060wm1Sc3XwrB/view?usp=sharing) into "out_2nd" folder.

    Run ```python test.py```

    The saliency maps will be saved in the "eval/maps" folder.

# Citation

*If you find this work is helpful, please cite our paper:
```
@InProceedings{PSOD_aaai2022,
  title={Weakly-Supervised Salient Object Detection Using Point Supervison},
  author={Gao, Shuyong and Zhang, Wei and Wang, Yan and Guo, Qianyu and Zhang, Chenglong and He, Yangji and Zhang, Wenqiang},
  booktitle={AAAI},
  year={2022}
}
```
