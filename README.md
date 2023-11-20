## The problem:

Identify building footprints within satellite and aerial images for evaluation of their structural integrity.

## Completed components:

- [X] Import satellite or aerial images from the designated data directory.
- [X] Conduct image preprocessing to optimise data quality.
- [X] Detect major types of objects such as buildings, cars or airplanes. --> CURRENTLY ON

## TODO:

- [ ] Employ detection techniques to identify the precise building footprints.
- [ ] Generate a GeoJSON file containing polygon representations of the detected building footprints.
- [ ] Create new images with the building footprints superimposed for visualisation.
- [ ] Obtain or generate tagged images of damaged buildings.
- [ ] Determine the presence or absence of damage to the buildings.
- [ ] Assess the degree of damage, categorising it on a scale ranging from 'none' to 'low,' 'medium,' 'high,' or 'full.'


## Plan of action:
The following points describe the steps I would take to complete the requirements for this task.

#### Employ detection techniques to identify the precise building footprints.
This point would involve selecting the building footprints from the outputs of the preceding step.

#### Generate a GeoJSON file containing polygon representations of the detected building footprints.
Post-Processing:

For each detected bounding box, the box would be converted to a polygon representation using libraries such as 
Shapely to help with geometric operations. Polygon representations would then be stored, along with any associated 
metadata.

Output Formatting:

With the polygons stored, the information would then be organised into a GeoJSON format, specifying the type (e.g., 
"FeatureCollection"). For each building footprint, I would create a GeoJSON feature with its polygon coordinates.


#### Create new images with the building footprints superimposed for visualisation.
Overlay Building Footprints:

To achieve this, I would load the original images and overlay the polygon representations of the building footprints 
onto the images. This would be done by drawing the polygons (numpy array) on top of the images using image processing libraries like
GeoPandas, OpenCV or Matplotlib.

Visualisation:

Following this I would display the newly created images to ensure that the building footprints align with the original 
images.

#### Obtain or generate tagged images of damaged buildings.
Dataset Expansion:

As no labelled images of damaged buildings were provided, I would need to obtain a dataset of images containing examples
of different damage levels. This could be done by searching for images of damaged buildings online (such as on natural 
disaster databases, Kaggle or UNOSAT) or by generating a dataset using an image annotation platform such as Labelbox.
Labels would have to be consistent and follow a binary classification scheme (e.g., 0 for no damage, 1 for damaged).

Integration:

Upon obtaining the dataset, I would integrate the damaged building dataset with the existing dataset e.g. using Pandas,
followed by cleaning and splitting.

#### Determine the presence or absence of damage to the buildings.
Inference:

This step would require a model trained on the labelled data to then allow it to make predictions on the dataset 
containing damaged buildings. The model would then be capable of distinguishing between damaged and undamaged buildings
and classifying them as such once fed the original images.

#### Assess the degree of damage, categorising it on a scale ranging from 'none' to 'low,' 'medium,' 'high,' or 'full.'
Categorisation:

Based on the model predictions, it would categorise the degree of damage for each building. I would establish a scale 
ranging from 'none' to 'low,' 'medium,' 'high,' or 'full' based on the model's confidence scores, based on the following
factors:
- The extent of the damage (e.g., the proportion of the building that is damaged)
- The type of damage (e.g., the type of damage to the building's structure)
- The location of the damage (e.g., the location of the damage on the building)


Validation:

I would validate the model's predictions against ground truth (or human-labeled data), adjusting the categorisation 
scale or fine-tuning the model if needed.