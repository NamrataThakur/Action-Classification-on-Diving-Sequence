# Automated Score Prediction for Diving Sequences

The repository contains code from:
- I3D Code Extraction: https://github.com/v-iashin/video_features

## Folder Structure
configs: I3D config *<br>
images: Output directory for intermediate processing<br>
modelcheckpoints: Contains the model flat files to be loaded on the server (e.g. temporal segmentation model, automated scoring model etc)<br>
models: I3D models, Repnet models *<br>
postman: Contains postman POST stub for testing the server<br>
uploads: Output directory for videos that were submitted in the POST request<br>
utils: I3D utils * <br>
* means it was part of the I3D repository code (not developed by us)<br>

## Environment Setup:
There are 2 environments to setup, 1 for the Repnet Flask Server, and 1 for the Backend Flask Server<br>
### Repnet Flask Server
`conda env create -n repnet --file repnet_env/environment_repnet_fromhist.yml python=3.6`<br>

### Main Backend Environment
`conda create -n backend python=3.9.13`<br>
`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`<br>
`pip install -r requirments_backend.txt`<br>
`ipython kernel install --user --name=backend` <br>


## To run:
### Repnet Flask Server
Within the main folder, run: <br>
`conda activate repnet `<br>
`python RepnetFlask.py`<br><br>
Repnet endpoint will be located at http://localhost:5001/sstwist <br>
You can HTTP POST to this endpoint using the postman stub.

### Main Backend Environment
Open "Backend.ipnyb" in Jupyter Notebook, select the 'backend' python environment, run all cells. This will activate the Flask server at the last cell.<br>
The server endpoint for video analysis will be located at http://localhost:5000/videoupload <br>
You can HTTP POST to this endpoint using the postman stub.
