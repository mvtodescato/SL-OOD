# Self-learning OOD detection (SL-OOD)
## From paper "Confidence-Based Zero-Shot Out-of-Distribution Detection via Self-Learning"

# Code files
-zero_shot.py
	
This is the code of our approach to perform zero-shot OOD detection. Use -h tag to see the help of the code. Remember to extract features of the dataset first using simple_features.py. 

-few_shot.py

This is the code of our approach to perform few-shot OOD detection. Use -h tag to see the help of the code.

-simple_features.py
	
In this code we extract features from the datasets using pre-trained models (vit_g14 by default).

-data_loaders.py

Code to load the datasets. 

#Obs

To use CLIP you need to download and install it from: https://github.com/openai/CLIP.

Cifar100 and Cifar10 are the easier datasets to test.
