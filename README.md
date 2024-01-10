# Visionbased Multipage Classifier

## Overview
This repository contains the code and documentation for my master's thesis project titled "Visuelle Klassifizierung mehrseitiger Dokumente ohne OCR". The project, developed in collaboration with lector.ai GmbH, explores the use of self-attention in image-based transformer architectures for multi-page document understanding, specifically for the separation and classification of document stacks.

## Contents
1. **multipage_classifier:** Contains the implementations of the three proposed architectures as well as the encoder and decoder modules.
2. **training:** Contains the training scripts and the implementations of the Lightning modules

## Usage Guide
1. **Training and Evaluation:**
   - To train one of the architectures, please customise the train.py according to your needs and implement your own data source

2. **Trained Architecture:**
   - To use one of the trained models, simply load it with the LightningModule class and refer to the desired checkpoint 

## Contact
For questions or suggestions, you can reach me at scheibe.johannes@t-online.de

---
