# PathNet: Path-selective Point Cloud Denoising
This is our implementation of PathNet

## Environment
* Python 3.5+
* PyTorch 1.3.1+
* CUDA and CuDNN (CUDA 10.1+ & CuDNN 7.5+)
* TensorboardX (2.0) if logging training info. 

## Installation

You can install via conda environment .yaml file

```bash
conda env create -f env.yml
conda activate pathnet
```

## Datasets

Download link: https://drive.google.com/drive/folders/xxxxxxxxxxxxx

Please extract `test_data.zip`, `train _data.hdf5` to `data` folder.

## Denoise

## Train
Use the script `train.py` to train a model in the our dataset (the trained model will be saved at `./log/path-denoise/model/checkpoints/best_model.pth`):
``` bash
cd PathNet
### First stage
python train.py --epoch 200 --use_random_path 1
### Second stage
python train.py --epoch 300 --use_random_path 0

```
## Test (The filtered results will be saved at `./data/results`)
``` bash
cd PathNet
python test.py
```

## Citation






