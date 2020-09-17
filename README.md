# Real Time Traffic Light Detection using Deep Learning (YOLOv3)



## <u>Table of Contents</u>

* [About](#About)
* [Progress and TODO](#Progress-and-TODO)
* [Download Trained Weights](#Download-Trained-Weights)
* [Get the Dataset](#Get-the-Dataset)
* [Steps to Train](#Steps-to-Train)
  * [Query on Ultralytics YOLOv3 img-size](#Query-on-Ultralytics-YOLOv3-img-size)
* [To Detect Using the Trained Model](#To-Detect-Using-the-Trained-Model)
* [References](#References)



## <u>About</u>

***This project aims to detect traffic light in real time using deep learning as a part of autonomous driving technology.***

* [Click on the following video to get a better idea about the project and predictions](https://www.youtube.com/watch?v=yy3XsMFKeSg&feature=youtu.be).

[![Prediction Video](https://github.com/sovit-123/Traffic-Light-Detection-Using-YOLOv3/blob/master/preview_images/vid_prev3.PNG?raw=true)](https://youtu.be/yy3XsMFKeSg)



## <u>Progress and TODO</u>

* **Implementation for all the traffic light types are done. But the final model is still being trained almost every day to make it better. Check the [Download Trained Weights](#Download-Trained-Weights) section to get your desired weight files and try the model on you system.**

- [x] Detecting red (circular) `stop` sign.
- [x] Detection green (circular) `go` sign.
- [x] Train on for night time detection => Working but not perfect. Better updates to come soon.
- [x] Detecting `warningLeft` sign.
- [x] Detecting `goLeft` sign.
- [x] Detecting `stopleft` sign.
- [x] Detecting `warning` sign.
- [ ] Carla support => **This one is a bit tricky.**



## <u>Download Trained Weights</u>

***Download the trained weights from [here](https://drive.google.com/drive/folders/1nGRGqw5KP6js9UbXDL5G99j_jYdKgdXl?usp=sharing).***

* `best_model_12.pt`: **Trained for 67 epochs on all the traffic signs. Current mAP is 0.919**



## <u>Get the Dataset</u>

This project uses the [LISA Traffic Light Dataset.](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset). Download the dataset from Kaggle [here](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset).



## <u>Steps to Train</u>

* **The current train/test split is 90/10. The input image size is 608x608. So, it might take a lot of time to train if you train on a nominal GPU. I have trained the model on Google Colab with Tesla T4 GPU/P100 GPU. One epoch took with all the classes around 1 hour on a Tesla T4 GPU. Also, check the `cfg` folder and files before training. You have to use the cfg files corresponding to the number of classes you are training on. If you want to change the number of classes to train on, then you have to change the cfg file too. The current model has been trained on all 6 classes, so, the cfg file is `yolov3-spp-6cls.cfg`.** 

* Prepare the data. **Please do take a look at the paths inside the `prepare_labels.py` file and change them according to your preference and convenience**.
  * `python prepare_labels.py`
* Create the train and validation text files (**Current train/validation split = 90/10**).
  * `python prepare_train_val.py`
* To train on your own system (The current [model](https://drive.google.com/drive/folders/1nGRGqw5KP6js9UbXDL5G99j_jYdKgdXl?usp=sharing) has been trained for 30 epochs.)
  * **To train from scratch**: `python train.py --data <your_data_folder>/traffic_light.data --batch 2 --cfg cfg/yolov3-spp-6cls.cfg --epochs 55 --weights "" --name from_scratch`
  * **Using COCO pretrained weights**: `python train.py --data <your_data_folder>/traffic_light.data --batch 4 --cfg cfg/yolov3-spp-6cls.cfg --epochs 55 --multi-scale --img-size 608 608 --weights weights/yolov3-spp-ultralytics.pt --name coco_pretrained`
  * **To resume training**: `python train.py --data <your_data_folder>/traffic_light.data --batch 2 --cfg cfg/yolov3-spp-6cls.cfg --epochs <num_epochs_must_be_greater_than_previous_training> --multi-scale --img-size 608 608 --resume --weights weights/<your_weight_file>.pt --name <name_to_be_saved_with>`

### [Query on Ultralytics YOLOv3 img-size](https://github.com/ultralytics/yolov3/issues/456).

* Short answer: The image size in `cfg` file is not used. Only python executables' argument parser `img-size` argument is used.



## <u>To Detect Using the Trained Model</u>

* **Download the [weights here](https://drive.google.com/drive/folders/1nGRGqw5KP6js9UbXDL5G99j_jYdKgdXl?usp=sharing) first, and paste them under the `weights` folder.**
  * `python detect.py --source <path_to_your_test_video_file> --view-img --weights weights/<your_weight_file_name>.pt --img-size 608`

 

## <u>References</u>

### Articles / Blogs / Tutorials

* [Recognizing Traffic Lights With Deep Learning.](https://www.freecodecamp.org/news/recognizing-traffic-lights-with-deep-learning-23dae23287cc/)
* [Self Driving Vehicles: Traffic Light Detection and Classification with TensorFlow Object Detection API.](https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62)

### Papers

* [Detecting Traffic Lights by Single Shot Detection.](https://arxiv.org/pdf/1805.02523.pdf)
* [A Hierarchical Deep Architecture and Mini-Batch Selection Method For Joint Traffic Sign and Light Detection.](https://arxiv.org/pdf/1806.07987v2.pdf)
* [Accurate traffic light detection using deep neural network with focal regression loss.](https://pdf.sciencedirectassets.com/271526/1-s2.0-S0262885619X00062/1-s2.0-S0262885619300538/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGJS6acKy%2Bn%2BogLTPASdUHm2kcAgzf%2BqPN9p8OeOtqjLAiEA%2F%2BXJIsDU4zTfeAt64IuxzWijoPZCAo8bGluHqWEyANsqvQMIuf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARADGgwwNTkwMDM1NDY4NjUiDDRiyVid6olIGdZwzyqRA10sNlWjy52x5aHLEkbyTlAwKwbhfH5gpZfQkY5ZnbhmzmOJAyj16Ij6x1D3cJL3XTMMT9Bj8TXdEOISOnDN2ZDThSTyotxowSzF3GN1V%2Brwgsv07x6GgyUGQz1TsZrbNxrdV2nYPKukv9PUNdcyDXeIWYh5emqvRSl75xtX5%2BGA9%2Be8OkAe8LjrsQJO4M%2BWL5vtSfc2ljzZH%2B%2FWHRwT8YJy8HWVoH1RyEOa1UdOaqfC1f2LYi2AiyAhEg4ODoAqrC9IXDOX%2BynMp4YbmUfUXff%2BCb%2F%2FpBfnuxYXXHGqZxFwf6hex%2FlQietzZ%2FJZnfM1dxZFkWdZjXMPeY6J6k5itnCQt6155HICBAaCD4jnCD93EG3CWTcQFGw5Fa59xkM6dRcyjFCyjvvOoDcOQkOdC9KkqXTEsviKA%2BGtfbR9VdfHxXTz6Eg3L2r0e%2FMD%2BWnKC9gE1O305BfGwVpH8QoC4y2YA6J6EB5SRcYcAYfVHEXae8jFcmT7RwqMlNmkvi5UARGyOOOj0HfuPQQj2Yn1c7qAMKKTk%2FoFOusBF61AXrHbnIYcGm4t9%2FshIODSgtKRGuw2AgBfRK8OQzmSoPfxhmZBph8Cg7vLOWlc6tygObNnLajEnuHOqENs0MNVERQRqeypLtugKOjYPTXhx6c2QHdu3dxq2xxVl4G%2FouOSad0Jk4shK1tvi4zBK7XubyhBnZg2nYEPJY87jCqMiyi8frITa51hPkILVTPH%2BMnWj71w52itNJCgoZ%2FLGKr%2F0yvE4ASCGEP0mGPdv3%2BkRJdQDNXnTlZZJ2jBDnUF8ppTA%2F5Ts8TG0MlXlvVmokNAHToumbuwlKA6LtGQFM5Ik3ksBZ4y2v3mMw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200825T092944Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZSN4AUAD%2F20200825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1a06167c3e97cae86c5f885091428f6313cd222846cba3196edfdd450e77f805&hash=42e81b760f319091bff8aa28f407c0be53b094e96dedd3e5895cf54cbcec3de6&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0262885619300538&tid=spdf-d78c15ef-4334-4615-9de5-b6e7a4fbcc3c&sid=9cbac0327e3d654a474b03703362e7cee4bdgxrqb&type=client)

### GitHub

* The YOLOv3 code has been take from the [Ultralytics YOLOv3](https://github.com/ultralytics/yolov3) repo and modified according to the use case.
* [TL-SSD: Detecting Traffic Lights by Single Shot Detection.](https://github.com/julimueller/tl_ssd)
* [Detecting Traffic Lights in Real-time with YOLOv3.](https://github.com/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset)

### Dataset

* [LISA Traffic Light Dataset.](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset)

### Image / Video Credits 

* **These may include links and citations for the data that I use for testing. You can also use these links to obtain the videos.**
* `video1.mp4`: https://www.youtube.com/watch?v=yJrW8werMUs.
* `video2.mp4`: https://www.youtube.com/watch?v=pU8ThDYZcCc.# Traffic-Light-Detection-Using-YOLOv3
* `video3.mp4`: https://www.youtube.com/watch?v=iS5sq9IELEo.
* `video4.mp4`: https://www.youtube.com/watch?v=GfWskqDjeTE.
* `video5.mp4`: https://www.youtube.com/watch?v=7HaJArMDKgI.
* `video6.mp4`: https://www.youtube.com/watch?v=NK_HNF1C8yA.
* `video7.mp4`: https://www.youtube.com/watch?v=w-W9esW3eqI.
* `video8.mp4`: https://www.youtube.com/watch?v=RPDYLA8Rh_M.
* `video9.mp4`: https://www.youtube.com/watch?v=imeV3Pm-ZLE.
