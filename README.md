# GRID - Geographic Reconnaissance for Infrastructure Detection

The goals of this project is to identify SDG&E assets and to quantify the ability to identify risks from damaged infrastructure. [DETR](https://github.com/facebookresearch/detr/) will be used to train an object detection model that will be able to identify power poles while Google Static Maps with satellite view will be used as the publicly accessible data source.

## Data Sources:
**Training images** are obtained using the [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static).

**Image annotations** in COCO json format will be needed for the model to train on. Training and validation annotations should be placed within the `data/annotations` directory and be named `image_annotations_train.json` and `image_annotations_val.json` respectively.
> [!NOTE]
> For the GRID data specifically, the annotations have already been added in the `data/annotations` directory.

## Setup

### Conda Environment
After cloning repository, navigate to root level and run:
```
conda env create -f environment.yml
```

### DETR Model
You must clone the DETR repository in order to train the model:
```
git clone https://github.com/woctezuma/detr.git
cd detr
git checkout finetune
cd ..
```

### Create Training/Validation Datasets and Prepare files to train model
After adding `images` folder to root directory and cloning the DETR repository, run:
```
python scripts/initialize.py
```
This will download the model's "base" and split the data into training/validation sets based on the COCO json annotations.

### Train the Model
In order to train the model, run the following:
```
python detr/main.py \
  --dataset_file "custom" \
  --coco_path "data" \
  --output_dir "outputs" \
  --resume "detr/detr-r50_no-class-head.pth" \
  --num_classes 2 \
  --epochs 50 \
  --device cuda
```
The parameters preceded by "--" may be modified accordingly such as the number of epochs to train for.
> [!IMPORTANT]
> A GPU is required to timely train the model.

After the model is finished training, output files will be saved to the `outputs` directory.

### Visualize Model Results
Run the cells within the `Model-Visualization.ipynb` file to display image examples of training/validation data as well as graphs on specfiic fields of interest.

### Use Model
Run `run_model.py` to run the object detection model on an image. Image output with bounding boxes will be saved in `model_results ` directory.

### Use Map Interface
Visit [https://github.com/papa-noel/map_user_interface] for instructions.

# Project Structure

```
├── data/
│   ├── annotations
│       ├── image_annotations_train.json
│       ├── image_annotations_val.json
│   ├── images
│       ├── image_data
│           ├── image_split
│               ├── train
│               ├── val
│       ├── model_results
│
├── outputs/              <- Output from model training (do not commit)
│
├── scripts/              <- Python scripts to run in command line
│
├── .gitignore            <- Git ignore file
│
├── Model-Visualization.ipynb  <- Jupyter Notebook for visualizing model results
│
├── run_model.py          <- Python script to run model on an image
│
├── environment.yml       <- Conda environment file
│
└── README.md             <- The top-level README for repo
```



