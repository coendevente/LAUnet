# LAUnet
Deep learning program for left atrial and scar segmentation in GE-MRI images.

## File and directory architecture
- artifical_data/: Generate artificial scar in images
- core/: Main scripts of the repository
  - architectures/: different network architectures
  - augmentations/: scripts for offline and online augmentation
  - helper_functions.py: helper functions with general functions that are useful through the repository
  - imshow_3D.py: function to show 3D images
  - predict.py: predict a single image using a trained network
  - settings.py: file with main settings object
  - test.py: test the trained network
  - train.py: train a network
- data_exploration/: files for exploration of the data
- procedures/: scripts that execute training and or testing procedures using files in the core directory
- visualisations/: script for visualising results