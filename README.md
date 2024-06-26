# SimFIR

<p>
    <a href='https://arxiv.org/pdf/2308.09040.pdf' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://simfir.doctrp.top:20443/' target="_blank"><img src='https://img.shields.io/badge/Online-Demo-green'></a>
</p>

The official code for “[SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning](https://arxiv.org/pdf/2308.09040.pdf)”, ICCV, 2023.

<img width="617" alt="image" src="https://github.com/fh2019ustc/SimFIR/assets/50725551/c85184da-7641-4f3a-b9b0-dbe89a6ab787">


## 🚀 Demo [(Link)](https://simfir.doctrp.top:20443/)
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.

![image](https://github.com/fh2019ustc/SimFIR/assets/50725551/1cb83f12-5f2e-4347-81af-ef68e4c0c468)


## Dataset

This repository contains the [dataset](https://pan.baidu.com/s/1tLgujwGmwYGcI0-IYmZ12w?pwd=4zbq) organized into two main directories:
- **images**: Contains distorted image files.
- **flow**: Stores the warping flow files.

Each of these directories further comprises two sub-directories:
- **train**: For training data.
- **test**: For testing data.

Click [here](https://pan.baidu.com/s/1tLgujwGmwYGcI0-IYmZ12w?pwd=4zbq) to access the dataset.

## Inference 
1. Put the [pre-trained model](https://drive.google.com/file/d/11YYPZ_z7-YoduHRdlMhy6u_km25f-Y8h/view?usp=sharing) to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Run the script and the rectified images are saved in `$ROOT/rectified/` by default.
    ```
    python inference.py
    ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{feng2023simfir,
  title={Simfir: A simple framework for fisheye image rectification with self-supervised representation learning},
  author={Feng, Hao and Wang, Wendi and Deng, Jiajun and Zhou, Wengang and Li, Li and Li, Houqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12418--12427},
  year={2023}
}
```
   
