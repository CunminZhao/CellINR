## CellINR: Implicitly Overcoming Photo-induced Artifacts in Fluorescence Microscopy 

**Motivations**: Fluorescence microscopy is often compromised by prolonged high intensity illumination which induces photobleaching effects that generate photo-induced artifacts and severely impair image continuity and detail recovery. To address this challenge, we propose the CellINR framework, a case-specific optimization approach based on implicit neural representation. The method employs blind convolution and structure amplification strategies to map 3D spatial coordinates into the high frequency domain, enabling precise modeling and high-accuracy reconstruction of cellular structures while effectively distinguishing true signals from artifacts. Experimental results demonstrate that CellINR significantly outperforms existing techniques in artifact removal and restoration of structural continuity, and for the first time, a paired cell imaging dataset is provided for evaluating reconstruction performance, thereby offering a solid foundation for subsequent quantitative analyses and biological research.


### Dependencies and Installation
- Python 3.11.0
- Pytorch 2.0.1


### Dataset
You can refer to the following links to download the datasets of our collected dataset in 
[figshare](https://drive.google.com/file/d/1zlPE-U97VnFDuS7vnrXZpY55ckxm8A3B/view?usp=drive_link).

### Training  
```
python train.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz"
```

### Inference  
```
python eval.py --config ./Config/config.yaml --save_path "./save" --file "./data/yourdata.nii.gz" 
```


