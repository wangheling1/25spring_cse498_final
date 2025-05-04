Project: Greenland Ice Layer Segmentation using Neural Operators  
Author: [Your Name]  

Description:  
This project contains source code for processing radar echograms, implementing deep learning models (CNN, FNO, Transformer), and evaluating various fusion strategies for internal ice layer segmentation in the Greenland ice sheet.

---

### File Descriptions

1. **cnn.py**  
   Defines CNN-based segmentation architectures and training routines for baseline experiments.

2. **data_show.py**  
   Visualization utilities for radargrams and annotated internal layers. Used to generate intermediate inspection plots.

3. **depth_distribution.py**  
   Analyzes the vertical distribution of internal layer annotations across the dataset.

4. **fusion_model.py**  
   Draft implementation of fusion-based models. Final versions of all models are included in the Jupyter notebook.

5. **height_ana.py**  
   Performs statistical analysis of layer heights to support preprocessing and normalization.

6. **ideal_distribution.py**  
   Analyzes the ideal image size and structure distribution for standardization.

7. **process.py**  
   Core preprocessing pipeline for radargrams and annotations. Handles cropping, resizing, normalization, and mask generation.

8. **cse498_final_test2.ipynb**  
   Main notebook for running the final experiments. Includes model training, evaluation, metric computation, and visualizations.

---

Note:  
Visualization images (.png) and the sample MATLAB file (.mat) are provided for reference and illustration only. They are not required for training or inference.  

To access the preprocessed dataset used in this project, please visit:  
**GitHub**: https://github.com/wangheling1  
or contact me via email: **hew221@lehigh.edu**
