# G2Net: Gravitational-Wave Detection Kaggle Competition

## About the Competition

This is the repository containing the code I developed for the [G2Net Gravitational-Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection). The goal of this competition was to find gravitational-wave(GW) signals in detector noise from the LIGO/VIRGO observatories. The data provided was comprised of synthetic injections of GW wave signals overlaid with noise. 

The idea behind the competition was to use deep learning to produce a computationally efficient method of gravitational wave detection. This is important to allow for online detection, which is essential for passing the information of a detection to other detectors such as radio telescopes to allow for the detection from multiple sources and thereby drastically increase the amount of information gained from each detected event (multi-messenger astronomy(MMA)).

Currently gravitational waves are detected using either Bayesian inference or matched filter search. Both of these techniques are very accurate, but also prohibitively computationally expensive. This makes the real-time detection necessary for multi-messenger astronomy impossible. Deep learning based approaches are interesting as they front-load a large part of the computational expense, enabling online detection.

## About My Solution

When I started working on the competition the dominant solution was and continued to be until the end of the competition:

1. Basic signal filtering
2. Perform a constant Q-transform on the data, turning it into an image representing the amplitude of different frequency bins against the time of the feature.
3. Training an image recognition convolutional neural network(CNN) on the data to find the frequency amplitude spikes resulting from gravitational waves.

This task scaled well with model complexity, making EfficientNetb7 the go-to choice.

My idea was based on the fact that this seemed rather wasteful, since 2D CNNs are very complex and require significant computational resources to train and infer with.

This is why I wanted to create a 1D solution, tailor-made for the problem. My proposed pipeline was:

1. Basic signal filtering
2. Use a encoder-decoder 1D network to reconstruct a possible signal without the noise to increase the signal to noise ratio(SNR) as much as possible. The current version of this network is a 1D CNN-LSTM model based on the architecture proposed in [this paper](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.064046).
3. Use another 1D CNN. The goal of this network was to detect the presence of a signal and in a later version determine the approximate parameters of the wave to be able to use matched filtering to calculate the signal-to-noise ratio of the event and calculate the statistical significance of the event.



## File Structure of the Repository

### Directories

* papers: literature used during the project
* models: pretrained models and a csv with model performance. The model directories include predictions on the test
* working: stable version of the code files
* experimental: "nightly" version of the code files

### Code Files

* `Infer.ipynb`: inference notebook; currently not in use since the inference functionality has been included at the end of `train.ipynb`

* `makeDataset.ipynb`: notebook used to prototype functionality for `makeDataset.py`

* `makeDataset.py`: python file for signal processing (whitening and bandpassing) and turning the [dataset](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data) into 24 `tfrec` files: 

  * 16 train-files, each containing 35,000 labelled samples
  * 8 test-files, each containing 28250 unlabelled samples

  **CAUTION**: this work is done concurrently, which means running the file creates 24 processes and uses a lot of ram, so changes to the code might be necessary. I will try to make this easier in the future.

  For this file to work, the unzipped [dataset](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data) should be in `/root(of repo)/input/g2net-gravitational-wave-detection/`.

* `model*.h5` these are pre-trained versions of the pre-filter. Unfortunately, I have so far not created a less cluttered way of storing them. This will be changed soon.

* `RandomDataset.ipynb`: creates a random sampling of raw noise files, that are available under `/root(of repo)/input/ggwp/output/<name of detector>/`. The file expects noise samples sampled at 4kHz from each of the detectors in the subdirectories [`/H1/`,`/L1/`,`/V1/`]. It looks for matching files and will only include them if the same time-slot is found for all three detectors.

* `tfrec_from_prefilter.ipynb`: is intended as a pipeline for creating `tfrec` files with the pretrained filter if that proves to be advantageous, but currently is used to analyze different pre-trained versions of the filter on different datasets

* `tfrecfromh5.ipynb`: this notebook is used to create `tfrec` files from synthetic datasets created using a version of the data generation code introduced in [this paper](https://arxiv.org/pdf/1904.08693.pdf), which allows for generating data of the Virgo detector in addition to the two LIGO detectors. It also includes some smaller changes and improvements.

* `train.ipynb`: the pipeline for training the detection model. This also includes an automatic inference on the test files. Requires `tfrec`-based datasets, which can be created using `makeDataset.py` 
* `train_filter.ipynb`: pipeline for training the filter

## Prerequisites

To run this project, you will need `Jupyter Notebooks` with a python 3 kernel to be installed on your machine. Instructions on how to install `Jupyter Notebooks` can be found on the [Jupyter website](https://jupyter.org/install).

The file structure expected by the notebook is modelled after the Kaggle file structure, so that the notebook can be run on both Kaggle and Windows: The root directory should contain two directories, input and working. Place the extracted folder of the [dataset](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data) in input and the notebook in the working directories, respectively. This way the code files will work without any changes to the code itself.