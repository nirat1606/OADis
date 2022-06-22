## Instructions

This code works for Compositional split of MIT-States. We will release the dataset VAW-CZSL, and code for UT_Zappos, upon acceptance. Although, this code is complete in covering our approach, OADis architecture is there in this code (named as oaclip), only hyperparameters will be slightly different for rest of the datasets.

To run the code:
python train.py --cfg config/mit-states.yml

Provide location paths for dataset and logs in the mit-states.yml

The code works well, and is tested for:
Pytorch - 1.6.0+cu92
Python - 3.6.12
tensorboardx - v2.4