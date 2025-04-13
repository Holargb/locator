Dataset

Mall Dataset http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html 

ShanghaiTech https://github.com/desenzhou/ShanghaiTechDataset?tab=readme-ov-file

Dataset Format

The options --dataset and --train-dir should point to a directory. 

This directory must contain your dataset, meaning: 

One file per image to analyze (png, jpg, jpeg, tiff or tif). 

One ground truth file called gt.csv with the following format: 

filename,count,locations 

img1.png,3,"[(28, 52), (58, 53), (135, 50)]" 

img2.png,2,"[(92, 47), (33, 82)]"

