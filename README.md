# PSOD (Point Saliency)
# Weakly-Supervised Salient Object Detection Using Point Supervision


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
![VIZ](https://user-images.githubusercontent.com/34783695/165979177-3ecef77f-d553-4747-97f7-2f7788c2adad.png)

* point supervised DUTS: [google](https://drive.google.com/file/d/1ZV2Bk1nZ3GRqcVvrabybSKT8N-1XsSH8/view?usp=sharing)

* training images and "train.txt" file: [google](https://drive.google.com/file/d/1jhHRB_GfJ4_Sm3ZgoTRbVSC7vCOwq_7r/view?usp=sharing)

# Download Relevant Necessary Data

* R50+ViT-B pretrained model: [google](https://drive.google.com/file/d/1N9zbAX97GRGnxz122A2W2wjcK-U8dX68/view?usp=sharing)

* edge maps: [google](https://drive.google.com/file/d/1Juzi-TZJfrB9iv_4UOYs60qn2VpZ033h/view?usp=sharing)

* gray image: [google](https://drive.google.com/file/d/11D_NY9UyunpPp19NqtFZvNpDCwtkCN3l/view?usp=sharing)


# Saliency map & Trained model


* trained model: [google](https://drive.google.com/file/d/1S8za3FiPalP0wRqazjj060wm1Sc3XwrB/view?usp=sharing)

* saliency maps: [google](https://drive.google.com/file/d/1TqIOXidkxkhq9nI0KBMApREam-EMnnr-/view?usp=sharing)

* initial pseudo-labels: [google](https://drive.google.com/file/d/1JbMHpTuMP6egfFRFNAYaiLJapAK80M8Q/view?usp=sharing)


# Train Model

1. Put the [R50+ViT-B pretrained model](https://drive.google.com/file/d/1N9zbAX97GRGnxz122A2W2wjcK-U8dX68/view?usp=sharing) into "models" folder.

2. Create the "dataset" folder in the main dirctory ("PSOD"). You should modify the ```create_link.sh``` file, then run it to create the soft link.

```
|----dataset
        |---edge
        |---gray
        |---image
        |---json
        |---train.txt
        |---filled_correct_img_gt  # Run "utills/EdgePoint2gt.py"
        |---filled_correct_mask  # Run "utills/EdgePoint2gt.py"
```



The [Point supervison](https://drive.google.com/file/d/1ZV2Bk1nZ3GRqcVvrabybSKT8N-1XsSH8/view?usp=sharing) (json) and [edge maps](https://drive.google.com/file/d/1Juzi-TZJfrB9iv_4UOYs60qn2VpZ033h/view?usp=sharing) (edge) are employed to generate [initial pseudo-labels](https://drive.google.com/file/d/1TqIOXidkxkhq9nI0KBMApREam-EMnnr-/view?usp=sharing) (we provided the initial pseudo-labels and you can use it directly for training, or you can run ```python utils/EdgePoint2gt.py```) to make initial pseudo-labels.

3. We organize all steps into a shell file, you can run ```bash edgepoint2gt_train_nss_retrain.sh``` to complete the entire training process.




# Test Model

1. Set the path of the test data in ```test.py```.

2. Create "out_2nd" folder and put the [trained model](https://drive.google.com/file/d/1S8za3FiPalP0wRqazjj060wm1Sc3XwrB/view?usp=sharing) into "out_2nd" folder.

    Run ```python test.py```

    The saliency maps will be saved in the "eval/maps" folder.

3. We also provied the final saliency maps saliency maps: [google](https://drive.google.com/file/d/1TqIOXidkxkhq9nI0KBMApREam-EMnnr-/view?usp=sharing).


# Evaluation Code

The "Eval_SingleMeth_MulDataset.py" in "Saliency-Evaluation-numpy" folder is used to evaluate the saliency maps.

# Citation

*If you find this work is helpful, please cite our paper:
```
@InProceedings{PSOD_aaai2022,
  title={Weakly-Supervised Salient Object Detection Using Point Supervision},
  author={Gao, Shuyong and Zhang, Wei and Wang, Yan and Guo, Qianyu and Zhang, Chenglong and He, Yangji and Zhang, Wenqiang},
  booktitle={AAAI},
  year={2022}
}
```

# Acknowledgement
[Weakly-Supervised Salient Object Detection via Scribble Annotations](https://github.com/JingZhang617/Scribble_Saliency)  
[Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence]()  
[Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers]()  

