# SimFIR
The official code for “SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning”, ICCV, 2023.

<img width="617" alt="image" src="https://github.com/fh2019ustc/SimFIR/assets/50725551/c85184da-7641-4f3a-b9b0-dbe89a6ab787">


> [SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning](https://arxiv.org/pdf/2308.09040.pdf) 
> ICCV 2023


## 🚀 Demo [(Link)]
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.
4. Our demo environment is based on a CPU infrastructure, and due to image transmission over the network, some display latency may be experienced.

[![Alt text](https://user-images.githubusercontent.com/50725551/232952015-15508ad6-e38c-475b-bf9e-91cb74bc5fea.png)](https://demo.doctrp.top/)


## Inference 
1. Put the pre-trained model to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Run the script and the rectified images are saved in `$ROOT/rectified/` by default.
    ```
    python inference.py
    ```
