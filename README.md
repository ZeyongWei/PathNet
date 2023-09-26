# PathNet: Path-selective Point Cloud Denoising
This is our implementation of PathNet

## Abstract

> Current point cloud denoising (PCD) models optimize single networks, trying to make their parameters adaptive to each point in a large pool of point clouds. Such a denoising network paradigm neglects that different points are often corrupted by different levels of noise and they may convey different geometric structures. Thus, the intricacy of both noise and geometry poses side effects including remnant noise, wrongly-smoothed edges, and distorted shape after denoising. We propose PathNet, a path-selective PCD paradigm based on reinforcement learning (RL). Unlike existing efforts, PathNet enables dynamic selection of the most appropriate denoising path for each point, best moving it onto its underlying surface. Besides the proposed framework of path-selective PCD for the first time, we have two more contributions. First, to leverage geometry expertise and benefit from training data, we propose a noise- and geometry-aware reward function to train the routing agent in RL. Second, the routing agent and the denoising network are trained jointly to avoid under- and over-smoothing. Extensive experiments show promising improvements of PathNet over its competitors, in terms of the effectiveness for removing different levels of noise and preserving multi-scale surface geometries. Furthermore, PathNet generalizes itself more smoothly to real scans than cutting-edge models.

## Environment
* Python >=3.5
* PyTorch >=1.3.1
* CUDA >= 10.1
* h5py
* numpy
* scipy
* tqdm
* TensorboardX 

## Installation

You can install via conda environment .yaml file.

```bash
conda env create -f env.yml
conda activate pathnet
```

## Datasets and model
We provide pretrained models and datasets [here](https://drive.google.com/drive/folders/1qaxpcqBGVK59HBfTTS68AoaqSWLcp9si?usp=sharing).

Please extract `test_data.zip`, `train _data.hdf5` to `./data` folder, `best_model.pth` to `./log/path-denoise/model/checkpoints/best_model.pth`.

## Denoise

## Train

Training will be public after acception.

## Test (The filtered results will be saved at `./data/results`)
``` bash
cd PathNet
python test.py
```

## Citation

## Acknowledgements
The repository is based on:
- [Path-Restore](https://github.com/yuke93/Path-Restore)
- [PU-GAN](https://liruihui.github.io/publication/PU-GAN/)
- [PU-Net](https://github.com/yulequan/PU-Net). 
- [Mesh Denoising via Cascaded Normal Regression](https://wang-ps.github.io/denoising.html)

The point clouds are visualized with [Easy3D](https://github.com/LiangliangNan/Easy3D).

We thank the authors for their great work！

## License

This project is open sourced under MIT license.
