# Mean-Shifted Contrastive Loss for Anomaly Detection
Official PyTorch implementation of [**“Mean-Shifted Contrastive Loss for Anomaly Detection”**](https://arxiv.org/pdf/2106.03844.pdf) (AAAI 2023).

## Virtual Environment
Use the following commands:
```
cd path-to-directory
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt
```

## Experiments
To replicate the results on CIFAR-10 for a specific normal class:
```
python main.py --dataset=cifar10 --label=n
```
Where n indicates the id of the normal class.

To replicate the results on CIFAR-10 with ResNet18 for a specific normal class:
```
python main.py --dataset=cifar10 --label=n --backbone=18
```
Where n indicates the id of the normal class.

Use the ```--angular``` flag to jointly optimize the mean-shifted contrastive loss and the angular center loss.  

To run experiments on different datasets, please set the path in utils.py to the desired dataset.

## Video Anomaly Detection
See our new paper [**“Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection”**](https://arxiv.org/pdf/2212.00789.pdf) which achieves state-of-the-art video anomaly detection performance on multiple benchmarks including 85.9% ROC-AUC on the ShanghaiTech dataset.

[**GitHub Repository**](https://github.com/talreiss/Accurate-Interpretable-VAD)

## Citation
If you find this useful, please cite our paper:
```
@inproceedings{reiss2023mean,
  title={Mean-shifted contrastive loss for anomaly detection},
  author={Reiss, Tal and Hoshen, Yedid},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={2155--2162},
  year={2023}
}
```
