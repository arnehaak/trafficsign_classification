# Traffic Sign Classification

## Preparation

Extract the content of "traffic_signs_train.zip" into directory "tsdata/train"!  
Extract the content of "traffic_signs_test.zip" into directory "tsdata/test"!

## Environment Setup

Tested with Python 3.7.7 (AMD64) on Windows 10 with Anaconda.

Environment Setup with Anaconda:

    conda create --name arnepy37 python=3.7 anaconda
    
    conda activate arnepy37
    
    conda install -c anaconda tensorflow-gpu
    conda install -c anaconda keras
    conda install -c conda-forge opencv
    
    # As of October 2020, there is an issue in Conda:
    # When conda install tensorflow it installs 2.1.0 but it brings with it
    # tensorflow-estimator 2.2.0. To fix this problem simply run the following
    # command after installing tensorflow 2.1.0 in Conda.
    # This advice is valid until conda switches to TF 2.2.0
    # https://stackoverflow.com/a/63218205
    conda install tensorflow-estimator==2.1.0
