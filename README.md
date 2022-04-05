# DeepEM
Deep-learning based particle picking program for single particle cryo-EM.  
This version is based on VGG (Very Deep Convolutional Networks), contributed by Yinping Ma.

Requirements:  
Python 3  
Tensorflow 1.9  
Mrcfile https://mrcfile.readthedocs.io/en/latest/index.html  
Other python packages mentioned in the code

Basic Usage:
Edit arg.py to change parameters  
Training: python train.py  
Picking: python predict.py  

Original version:  
http://ipccsb.dfci.harvard.edu/deepem/index.html

Reference:  
Y. Zhu, Q. Ouyang, Y. Mao. A deep convolutional neural network approach to single-particle recognition in cryo-electron microscopy. BMC Bioinformatics 18, 348 (2017). Learn More. arXiv: 1605.05543.
