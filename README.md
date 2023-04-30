# Automated Score Prediction for Diving Sequences

The repository contains code from:
- I3D Code Extraction: https://github.com/v-iashin/video_features

## Folder Structure
configs: I3D config * 
images: Output directory for intermediate processing
modelcheckpoints: Contains the model files to be loaded on the server (e.g. temporal segmentation model, automated scoring model etc)
models: I3D models *
postman: Contains postman POST stub for testing the server
uploads: Output directory for videos that were submitted in the POST request
utils: I3D utils *
* means it was part of the I3D repository code (not developed by us)

## To run:
Open "Backend.ipnyb" in Jupyter Notebook, run all cells. This will activate the Flask server at the last cell.
The server endpoint for video analysis will be located at http://localhost:5000/videoupload
You can HTTP POST to this endpoint using the postman stub.
