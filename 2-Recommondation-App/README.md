# ALS Recommendation API

1. This is a "Ready-to-deploy recommendation system" using ALS (Alternating Least Squares) and FastAPI.
2. Clients can query top-N recommendations for a given user via an API.


## Features

1. Train an ALS model on implicit feedback data (`interactions.csv`).
2. Hyperparameter tuning for `factors`, `regularization`, and 'iterations'.
3. Save the trained model and mappings.
4. FastAPI app provides `/recommend/` endpoint for querying recommendations.
5. Swagger UI for easy testing at `/docs`.


## Requirements

1. Python 3.10+
2.  Conda (Anaconda or Miniconda recommended)



## Setup & Running the Project
### 1. Create Conda Environment
conda create -n reco_env python=3.10 -y
conda activate reco_env
pip install -r requirements.txt


# Folder Structure
recommendationApp/
├─ data/
│  └─ interactions.csv
├─ model.pkl
├─ mappings.pkl
├─ app.py
├─ requirements.txt
└─ README.md


# Valid User_Ids
{"user_ids":[26,68,104,93,143,24,73,90,111,43,137,69,128,136,1,76,56,7,20,45,70,57,113,35,39,5,138,88,84,2,145,119,94,72,11,36,62,150,101,124,34,31,74,131,147,22,120,48,17,13,108,77,125,21,54,144,60,49,44,55,40,132,141,80,30,118,27,65,135,107,117,18,6,50,106,121,86,52,122,110,42,83,64,134,78,33,46,97,96,10,14,32,59,28,9,8,129,47,12,41,37,79,127,25,92,126,95,109,114,123,98,19,71,103,89,105,38,87,16,149,100,67,140,82,142,130,85,112,63,58,29,99,115,15,148,81,75,3,146,139,91,66,51,53,102,4,133,116,23,61]}

