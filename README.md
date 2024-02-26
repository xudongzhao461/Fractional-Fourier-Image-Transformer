# Fractional-Fourier-Image-Transformer： Fractional Fourier Image Transformer for Multimodal Remote Sensing Data Classification

This example implements the paper [Fractional Fourier Image Transformer for Multimodal Remote Sensing Data Classification]

## Usage

### Data set links

1. Houston dataset were introduced for the 2013 IEEE GRSS Data Fusion contest. Data set links comes from http://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/

2. The authors would like to thank Dr. P. Ghamisi for providing the Trento Data. 

3. The MUUFL Gulfport Hyperspectral and LIDAR Data [1][2] is Available from https://github.com/GatorSense/MUUFLGulfport/.

[1] P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, “MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set,” University of Florida, Gainesville, FL, Tech. Rep. REP-2013-570, Oct. 2013.

[2] X. Du and A. Zare, “Technical Report: Scene Label Ground Truth Map for MUUFL Gulfport Data Set,” University of Florida, Gainesville, FL, Tech. Rep. 20170417, Apr. 2017. Available: http://ufdc.ufl.edu/IR00009711/00001.

### Training

Train the HSI and LiDAR-based DSM
```
python FrIT_main.py 
```

## Results
More details can be found in the paper.

## Citation

Please kindly cite the papers if this code is useful and helpful for your research.

X. Zhao et al., "Fractional Fourier Image Transformer for Multimodal Remote Sensing Data Classification," in IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 2, pp. 2314-2326, Feb. 2024, doi: 10.1109/TNNLS.2022.3189994.

```
@ARTICLE{9830635,
  author={Zhao, Xudong and Zhang, Mengmeng and Tao, Ran and Li, Wei and Liao, Wenzhi and Tian, Lianfang and Philips, Wilfried},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Fractional Fourier Image Transformer for Multimodal Remote Sensing Data Classification}, 
  year={2024},
  volume={35},
  number={2},
  pages={2314-2326},
  keywords={Feature extraction;Transformers;Laser radar;Data mining;Discrete Fourier transforms;Visualization;Semantics;Fractional Fourier image transformer (FrIT);hyperspectral image (HSI);light detection and ranging (LiDAR);multimodal data},
  doi={10.1109/TNNLS.2022.3189994}}


```


