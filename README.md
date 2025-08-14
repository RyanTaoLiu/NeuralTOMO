# [Neural Co-Optimization of Structural Topology, Manufacturable Layers, and Path Orientations for Fiber​-Reinforced Composites](https://ryantaoliu.github.io/NeuralTOMO/)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://ryantaoliu.github.io/NeuralTOMO/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)

**Authors:**  
[Tao Liu\*](#), [Tianyu Zhang\*](https://zhangty019.github.io/), Yongxue Chen, Weiming Wang,  Yu Jiang, Yuming Huang, and [Charlie C.L. Wang†](https://mewangcl.github.io/)

---
## 📖 Overview
We propose a **neural network-based computational framework** for the **simultaneous optimization** of:

- **Structural topology**
- **Curved manufacturing layers**
- **Path orientations**

Our method targets **fiber-reinforced thermoplastic composites**, aiming to **maximize anisotropic strength** while preserving manufacturability for **filament-based multi-axis 3D printing**.

**Key features:**
- Three **implicit neural fields** to represent **geometry**, **layer sequence**, and **fiber orientation**.
- Integrated and differentiable objectives:Anisotropic strength, Structural volume,Machine motion control, Layer curvature and thickness
- **Physical validation**: up to **33.1% improvement** in failure loads over sequential optimization.

<p align="center">
  <img src="https://ryantaoliu.github.io/NeuralTOMO/images/2.jpg" width="80%">
</p>

*Figure: Overview of our slicing framework for co-optimization of design and manufacturing constraints in multi-axis 3D printing.*

## 🚀 Installation
### Git download the code
```bash
git clone https://github.com/RyanTaoLiu/NeuralTOMO
cd NeuralTOMO
mkdir data
```

### Create and activate the Python environment
```
conda create -n NeuralTO python=3.10
conda activate NeuralTO
```

### Install dependencies
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
conda install -c conda-forge scikit-sparse

pip install comet_ml paramiko matplotlib ninja pyvista
pip install mkl py_MKL
pip install diso=0.1.2
```

### 🧪Example of TipCantilever
```
python main.py \
  --problem=TipCantilever_30_20_20_midLoad \
  --material=PLAPlus \
  --no-isotropic \
  --wSF 0 \
  --wSR 1 \
  --wStress 0 \
  --desireVolumeFraction 0.15 \
  --wHarmonic 1 \
  --numLayers 5 \
  --numNeuronsPerLayer 128 \
  --fourierMap \
  --learningRate 0.001
```
Results will be saved in *./data*.

```
python main.py --help
```
Will show the help documents for args.

### 📂Data structure & Notes
- **Boundary condition**, See [*'./settings/problems/testproblem.py'*](https://github.com/RyanTaoLiu/NeuralTOMO/blob/main/settings/problems/testProblem.py). 
Based on the paper, **An efficient 3D topology optimization code written in Matlab**\[1\]. Add any new boundary condition as the new py file follow the *testproblem.py* and can be called by the arg *'--problem'*.

- Voigt tensor order: ```[xx, yy, zz, yz, xz, zy]```

- Two rotation matrix quasi-isotropy \[2\](https://doi.org/10.1007/s00158-023-03586-w) and anisotropy\[3\](https://doi.org/10.1007/s00158-019-02461-x), 

## Reference
- [1] K. Liu and A. Tovar. **An efficient 3D topology optimization code written in Matlab**. *Structural and Multidisciplinary Optimization*, **50**(6):1175–1196, 2014. [https://doi.org/10.1007/s00158-014-1107-x](https://doi.org/10.1007/s00158-014-1107-x)  

- [2] Y.R. Luo, R. Hewson, and M. Santer. **Spatially optimised fibre-reinforced composites with isosurface-controlled additive manufacturing constraints**. *Structural and Multidisciplinary Optimization*, **66**, 130, 2023. [https://doi.org/10.1007/s00158-023-03586-w](https://doi.org/10.1007/s00158-023-03586-w)  

- [3] D.R. Jantos, K. Hackl, and P. Junker. **Topology optimization with anisotropic materials, including a filter to smooth fiber pathways**. *Structural and Multidisciplinary Optimization*, **61**:2135–2154, 2020. [https://doi.org/10.1007/s00158-019-02461-x](https://doi.org/10.1007/s00158-019-02461-x)  

- [4] T. Liu, T. Zhang, Y. Chen, Y. Huang, and C.C.L. Wang. **Neural Slicer for Multi-Axis 3D Printing**. *ACM Transactions on Graphics*, **43**(4), Article 85, 15 pages, July 2024. [https://doi.org/10.1145/3658212](https://doi.org/10.1145/3658212)  

- [5] T. Liu, T. Zhang, Y. Chen, W. Wang, Y. Jiang, Y. Huang, and C.C.L. Wang. **Neural Co-Optimization of Structural Topology, Manufacturable Layers, and Path Orientations for Fiber-Reinforced Composites**. *ACM Transactions on Graphics*, **44**(4), Article 128, 17 pages, August 2025. [https://doi.org/10.1145/3730922](https://doi.org/10.1145/3730922)  



## Contact information: 
- Tao Liu  (tao.liu@manchester.ac.uk)
- Tianyu Zhang  (tianyu.zhang@manchester.ac.uk)
- Charlie C.L. Wang  (changling.wang@manchester.ac.uk)





