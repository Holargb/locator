Dataset

Mall Dataset http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
ShanghaiTech https://github.com/desenzhou/ShanghaiTechDataset?tab=readme-ov-file

Dataset Format 

The options --dataset and --train-dir should point to a directory. This directory must contain your dataset, meaning:
One file per image to analyze (png, jpg, jpeg, tiff or tif). 
One ground truth file called gt.csv with the following format: 
filename,count,locations 
img1.png,3,"[(28, 52), (58, 53), (135, 50)]" 
img2.png,2,"[(92, 47), (33, 82)]"

Environment 
name: object-locator
channels:
    - pytorch
    - conda-forge
    - defaults
dependencies:
    - imageio=2.3.0
    - ipdb=0.11
    - ipython=6.3.1
    - ipython_genutils=0.2.0
    - matplotlib=2.2.2
    - numpy=1.14.3 # install after having installed scikit
    - opencv=3.4.1 # pip install opencv-contrib-python==3.4.1.15
    - pandas=0.22.0
    - parse=1.8.2
    - pip=9.0.3
    - python=3.6.5
    - python-dateutil=2.7.2
    - scikit-image=0.13.1
    - scikit-learn=0.19.1
    - scipy=1.0.1
    - setuptools=39.1.0
    - tqdm=4.23.1
    - xmltodict=0.11.0
    - pytorch=1.0.0 # pip install torch==1.0.1 torchvision==0.2.2
    - pip:
        - ballpark==1.4.0
        - visdom==0.1.8.5
        - peterpy
        - torchvision==0.2.1

