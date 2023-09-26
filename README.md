# SimFIR
The official code for “[SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning](https://arxiv.org/pdf/2308.09040.pdf)”, ICCV, 2023.

<img width="617" alt="image" src="https://github.com/fh2019ustc/SimFIR/assets/50725551/c85184da-7641-4f3a-b9b0-dbe89a6ab787">


## 🚀 Demo [(Link)](https://simfir.doctrp.top:20443/)
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.

![image](https://github.com/fh2019ustc/SimFIR/assets/50725551/1cb83f12-5f2e-4347-81af-ef68e4c0c468)


## Dataset


## Inference 
1. Put the pre-trained model to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Run the script and the rectified images are saved in `$ROOT/rectified/` by default.
    ```
    python inference.py
    ```
   
