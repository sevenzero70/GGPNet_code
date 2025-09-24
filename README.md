# GGPNet_code
This repository contains the official implementation of our paper:  
**"Unlocking spatial textures: Gradient-guided pansharpening for enhancing multispectral imagery"**.
The code is being organized! 🚀🚀🚀

## Environment
The code is developed and tested on Ubuntu 18.04 with Python 3.8.5. The following packages are required:
> pip install -r requirements.txt

## Datasets
The code is tested on the following datasets:
- [PanCollection](https://github.com/liangjiandeng/PanCollection)

## Training
To train the model, run the following command:
> python train_PReNet.py

## Testing
To test the model, run the following command:
> python test_s_sota.py --model_name PReNetGradient

## Pretrained models
We will provide the following pretrained models

## Citation
Liang L, Li T, Wang G, et al. Unlocking spatial textures: Gradient-guided pansharpening for enhancing multispectral imagery[J]. Neurocomputing, 2025: 131607.

If you find our work useful, please consider citing our paper.
> @article{liang2025unlocking,
  title={Unlocking spatial textures: Gradient-guided pansharpening for enhancing multispectral imagery},
  author={Liang, Lanyue and Li, Tianyu and Wang, Guoqing and Mei, Lin and Tang, Xiongxin and Qiao, Chaofan and Xie, Dongyu},
  journal={Neurocomputing},
  pages={131607},
  year={2025},
  publisher={Elsevier}
}
