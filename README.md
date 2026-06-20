SSD Implementation in Pytorch
========

This repository implements SSD, with training, inference and mAP evaluation in PyTorch.
Most of the code is just parts of pytorch ssd implementation and all I have done is gotten rid of abstractions and commented the code.

The repo provides code to train on voc dataset. Specifically I trained on trainval images of VOC 2007 dataset and for testing, I use VOC2007 test set.

| model | c(t) sec/pixel | K(t) sec | τ pixels |
| --- | --- | --- | --- |
| roissd | 0.00000155 | 0.02155055 | 13883.8 |
| yolo | 0.00000013 | 0.00858294 | 66691.3 |
|  |  |  |  |

**Таблиця 2. Параметри лінійної моделі вартості та поріг злиття \( \tau = K_t / c_t \) на різних пристроях для ROI-SSD та YOLO**

| Модель | Система | Режим | Пристрій | \(c_t\) (sec/pixel) | \(K_t\) (sec) | \(\tau\) (pixels) |
|:---:|:---:|:---:|---|---:|---:|---:|
| RoiSSD | A | — | GPU RTX 3060 Ti 8GB | \(8.0 \cdot 10^{-8}\) | 0.018254 | 225,677 |
| RoiSSD | A | — | CPU i7-11700K | \(1.48 \cdot 10^{-6}\) | 0.0256 | 17,295 |
| RoiSSD | B | — | GPU RTX 3060 12GB | \(7.0 \cdot 10^{-8}\) | 0.013174 | 185,870 |
| RoiSSD | B | — | CPU i9-9900KF | \(1.70 \cdot 10^{-6}\) | 0.011117 | 6,549–7,679 |
| RoiSSD | C | Plugged | GPU RTX 4050 Laptop | \(8.0 \cdot 10^{-8}\) | 0.008102 | 96,216 |
| RoiSSD | C | Plugged | CPU Core 5 210H | \(2.16 \cdot 10^{-6}\) | 0.010341 | 4,778 |
| RoiSSD | D | Energy saving | GPU RTX 4050 Laptop | \(5.0 \cdot 10^{-8}\) | 0.018075 | 400,144 |
| RoiSSD | D | Energy saving | CPU i5 H210 | \(3.85 \cdot 10^{-6}\) | 0.012567 | 3,268 |
| RoiSSD | E | — | CPU Raspberry Pi 5 4GB | \(1.328 \cdot 10^{-5}\) | 0.030140 | 2,269 (≈1300–3500) |
| YOLO26n | A | — | GPU RTX 3060 Ti 8GB | \(1.0 \cdot 10^{-8}\) | 0.0189 | 3,920,311 |
| YOLO26n | A | — | CPU i7-11700K | \(1.66 \cdot 10^{-7}\) | 0.0150 | 105,006 |
| YOLO26n | E | — |  CPU Raspberry Pi 5 4GB | \(0.81 \cdot 10^{-6}\) | 0.0186 | 22,935 |


## SSD Explanation and Implementation Video
<a href="https://youtu.be/c_nEue9itwg">
   <img alt="SSD Explanation and Implementation" src="https://github.com/user-attachments/assets/663754cf-93a7-4b7a-9a0f-ff094f73e90a" width="400">
</a>


## Result by training SSD on VOC 2007 dataset 
One should be able to get **71-72% mAP** by training on VOC 2007 trainval images(**68% reported in paper**).

Adding 2012 trainval we should be able to get **>77% mAP**

<img src="https://github.com/user-attachments/assets/e21e3344-a0b7-4c91-b06d-6b83f62df0b0" width="250">
<img src="https://github.com/user-attachments/assets/0d128c3e-d4ab-4335-a18f-77b7553f9634" width="250">
<img src="https://github.com/user-attachments/assets/1c588ab8-975e-4ece-bb2e-679d6b9fb18d" width="250">
</br>

Here's an evaluation result that I got after training 100 epochs.
```
Class Wise Average Precisions
AP for class aeroplane = 0.7552
AP for class bicycle = 0.8384
AP for class bird = 0.7025
AP for class boat = 0.6543
AP for class bottle = 0.3411
AP for class bus = 0.8355
AP for class car = 0.8611
AP for class cat = 0.8682
AP for class chair = 0.4798
AP for class cow = 0.7453
AP for class diningtable = 0.7092
AP for class dog = 0.8582
AP for class horse = 0.8506
AP for class motorbike = 0.8259
AP for class person = 0.7721
AP for class pottedplant = 0.3939
AP for class sheep = 0.7300
AP for class sofa = 0.7626
AP for class train = 0.8615
AP for class tvmonitor = 0.7260
Mean Average Precision : 0.7286
```


## Data preparation
For setting up the VOC 2007 dataset:
* Create a data directory inside SSD-Pytorch
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the `VOC2007` directory inside `data` directory
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2007` directory and name it as `VOC2007-test` directory inside `data`
* If you want to use 2012 trainval images as well, then download VOC 2012 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2012` directory inside `data`
  * Ensure to place all the directories inside the data folder of repo according to below structure
      ```
      SSD-Pytorch
          -> data
              -> VOC2007
                  -> JPEGImages
                  -> Annotations
                  -> ImageSets
              -> VOC2007-test
                  -> JPEGImages
                  -> Annotations
              -> VOC2012 (if needed)
                  -> JPEGImages
                  -> Annotations
                  -> ImageSets
          -> tools
              -> train.py
              -> infer.py
          -> config
              -> voc.yaml
          -> model
              -> ssd.py 
          -> dataset
              -> voc.py
      ```

## ImageNet-VID VOC-Compatible Subset
If you want to benchmark VOC-trained models on ImageNet-VID without retraining, the repo now includes a standalone subset builder that creates a neighboring `VOC10KAnnotations` tree containing only selected clips and only VOC-overlap objects.

The script does not copy images. It copies and prunes XML files only, preserving the original train or val annotation structure so the existing ImageNet-VID dataset loader can discover selected frames by annotation presence.

Run the subset builder with:

```powershell
python -m tools.build_imagenet_vid_voc_subset \
    --data-root "D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Data\VID" \
    --ann-root "D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Annotations\VID" \
    --split val \
    --target-clips 200 \
    --clip-length 100 \
    --clip-stride 100 \
    --max-clips-per-video 3 \
    --overwrite
```

Key behavior:
* Only frames with at least one VOC-compatible ImageNet-VID object are eligible.
* Selected clips are contiguous in original frame indices.
* Written XMLs are pruned to VOC-overlap objects only.
* The output folder contains `manifest.json` and `selected_clips.csv` so the subset can be inspected and regenerated reproducibly.

The generated annotation layout looks like:

```
...\Annotations\VID
        -> train
        -> val
        -> VOC10KAnnotations
                -> train
                        -> a\video_name\frame.xml
                        -> b\video_name\frame.xml
                -> val
                        -> video_name\frame.xml
                -> manifest.json
                -> selected_clips.csv
```

To use the generated subset with the existing ImageNet-VID loader, point the annotation root in your config at the generated split directory. For example:

```yaml
dataset_params:
    train_data_root: 'D:\ImageNet-VID\...\Data\VID\train'
    train_ann_root: 'D:\ImageNet-VID\...\Annotations\VID\VOC10KAnnotations\train'
    test_data_root: 'D:\ImageNet-VID\...\Data\VID\val'
    test_ann_root: 'D:\ImageNet-VID\...\Annotations\VID\VOC10KAnnotations\val'
    filter_voc_overlap: true
```

This works because `dataset/imagenet_vid.py` already walks the image tree and only keeps frames whose XML exists under the configured annotation root.

## For training on your own dataset

* Update the path for `train_im_sets`, `test_im_sets` in config
* If you want to train on 2007+2012 trainval then have `train_im_sets` as `['data/VOC2007', 'data/VOC2012'] `
* Modify dataset file `dataset/voc.py` to load images and annotations accordingly specifically `load_images_and_anns` method
* Update the class list of your dataset in the dataset file.
* Dataset class should return the following:
    ```
  im_tensor(C x H x W) , 
  target{
        'bboxes': Number of Gts x 4 (this is in x1y1x2y2 format normalized from 0-1)
        'labels': Number of Gts,
        'difficult': Number of Gts,
        }
  file_path
  ```


## For modifications 
* In case you have GPU which does not support large batch size, you can use a smaller batch size like 2 and then have `acc_steps` in config set as 4(to mimic 8 batch size training).
* For using a different backbone you would have to change the following:
  * Change the backbone, extra conv layers and creation of feature maps in initialization of SSD model
  * Ensure the `out_channels` is correctly set as the channels in all feature maps to be used for prediction [here](https://github.com/explainingai-code/SSD-PyTorch/blob/main/model/ssd.py#L316)
  * In the forward method call the backbone and extra conv layers and ensure `outputs` is correctly set as list of feature maps [here](https://github.com/explainingai-code/SSD-PyTorch/blob/main/model/ssd.py#L472)

# Quickstart
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/explainingai-code/SSD-PyTorch.git```
* ```cd SSD-PyTorch```
* ```pip install -r requirements.txt```
* For training/inference use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train``` for training SSD on VOC dataset
* ```python -m tools.infer --evaluate False --infer_samples True``` for generating inference predictions
* ```python -m tools.infer --evaluate True --infer_samples False``` for evaluating on test dataset
* ```python -m tools.infer --evaluate True --infer_samples False --eval-mode default``` to evaluate once on the dataset transform defined in config (`dataset_params.transform_name`)
* ```python -m tools.infer --evaluate True --infer_samples False --eval-mode pad-loop``` to run fixed-padding sweep evaluation (`fixed_padding_roi_crop_{X}` or `fixed_padding_roi_crop_yolo_{X}`)
* ```python -m tools.train --final-eval-mode pad-loop``` to train with per-epoch intermediate default-transform mAP and a final pad-loop evaluation

### Inference/Evaluation modes
`tools/infer.py` supports:
* `--eval-mode default`: single evaluation pass using config transform.
* `--eval-mode pad-loop`: multi-run padding sweep from `0..200` with step `10`.

During training, `tools/train.py` now performs:
* intermediate mAP evaluation at the end of each epoch using `default` mode.
* final post-training evaluation using `--final-eval-mode` (`default` or `pad-loop`).

## Configuration
* ```config/voc.yaml``` - Allows you to play with different components of SSD on voc dataset  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of `task_name` key in config will be created

During training of SSD the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory
* Per-epoch unified metrics CSV at ```task_name/training_metrics.csv``` with columns:
    * `epoch`
    * `classification_loss`
    * `detection_loss`
    * `learning_rate`
    * `mAP`
    * `mean_detector_recall`

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples```

## Citations
```
@article{DBLP:journals/corr/LiuAESR15,
  author       = {Wei Liu and
                  Dragomir Anguelov and
                  Dumitru Erhan and
                  Christian Szegedy and
                  Scott E. Reed and
                  Cheng{-}Yang Fu and
                  Alexander C. Berg},
  title        = {{SSD:} Single Shot MultiBox Detector},
  journal      = {CoRR},
  volume       = {abs/1512.02325},
  year         = {2015},
  url          = {http://arxiv.org/abs/1512.02325},
  eprinttype    = {arXiv},
  eprint       = {1512.02325},
  timestamp    = {Wed, 12 Feb 2020 08:32:49 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/LiuAESR15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
