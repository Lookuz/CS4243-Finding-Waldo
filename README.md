# CS4243 Project: Find The Waldo

#### Prerequisites
Check the `requirements.txt` for the envrionment required. Python 3.7.3 is used.

`.xml` and `.jpg` files are not included. To run the notebook, you need to put the images (`000.jpg`...`079.jpg`) under `datasets/JPEGImages` and the ground truth annotation files (`000.xml`...`079.xml`) under `datasets/Annotations`

You can run the `Evaluation.ipynb` notebook to check detection on the validation dataset.

You can run the `SIFT_SURF Workflow.ipynb` notebook for a step-by-step demonstration of our method.

#### Structure
- data pipeline:
`dataset.py`, `utils.py`

- classifier training, model saving and loading:
`classification.py`

- feature extraction and vocabulary construction:
`descriptors.py`, `template.py`, `BagOfWords.py`

- sliding window detection:
`SlidingWindow.py`, `BagOfWords.py`, `detection.py`

- vocabulary:
`akaze_dict/*`, `bovw/*`, `complex_vocabs/*`

- models:
`models/*`, `complex_models/*` 

#### Authors:
- Choong Wey Yeh e0176617@u.nus.edu
- Nguyen Duy Son e0072469@u.nus.edu
- Yang Sihan e0248120@u.nus.edu
