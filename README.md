# kfr-python
An unofficial python implementation of using Kernel Foveated Rendering for image compression.

See the official resources for more information:<br>
[Website](https://xiaoxumeng1993.wixsite.com/xiaoxumeng/kernel-foveated-rendering)
[Paper](https://dl.acm.org/doi/10.1145/3203199)
[Shadertoy](https://www.shadertoy.com/view/lsdfWn)

## Usage

1. Install Python with PyTorch, Numpy, Matplotlib, Pillow
2. Create a file `image.png`
3. Run `python main.py` to generate a foveated image.

## Citation
Please cite their paper if you use this code in your research.
```bibtex
@article{10.1145/3203199,
author = {Meng, Xiaoxu and Du, Ruofei and Zwicker, Matthias and Varshney, Amitabh},
title = {Kernel Foveated Rendering},
year = {2018},
issue_date = {July 2018},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {1},
number = {1},
url = {https://doi.org/10.1145/3203199},
doi = {10.1145/3203199},
journal = {Proc. ACM Comput. Graph. Interact. Tech.},
month = {jul},
articleno = {5},
numpages = {20},
keywords = {foveated rendering, eye-tracking, log-polar mapping, virtual reality, head-mounted displays, perception}
}
```