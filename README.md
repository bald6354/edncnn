# Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras
Contains a MATLAB implementation for our [paper](https://arxiv.org/abs/2003.08282).  If you find this code useful in your research, please consider citing:

    @article{baldwin2020event,
      title={Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras},
      author={Baldwin, R and Almatrafi, Mohammed and Asari, Vijayan and Hirakawa, Keigo},
      journal={arXiv preprint arXiv:2003.08282},
      year={2020}
    }

This code was tested on an Ubuntu 18.04 system (i7-8700 CPU, 64GB RAM, and GeForce RTX 2080Ti GPU) running MATLAB 2019b. Matlab's image processing and computer vision toolboxes are required.

![Missing Image](https://github.com/bald6354/edncnn/blob/master/pics/1156-teaser.gif "Denoised Dataset")

## DVSNOISE20 Dataset (optional)
To download the dataset use: [dvsnoise20](https://sites.google.com/a/udayton.edu/issl/software/dataset).

## Reading AEDAT data into MATLAB (optional)
To read AEDAT (jAER) data into MATLAB use: [AedatTools](https://gitlab.com/inivation/AedatTools) or [AedatTools Alt](https://github.com/simbamford/AedatTools/).

## Reading AEDAT4 data into MATLAB (optional)
To read AEDAT4 (DV) data into MATLAB use: [aedat4tomat](https://github.com/bald6354/aedat4tomat).
