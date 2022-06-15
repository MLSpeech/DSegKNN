# DSegKNN: Unsupervised Word Segmentation using K Nearest Neighbors

Tzeviya Sylvia Fuchs (fuchstz@cs.biu.ac.il) \
Yedid Hoshen (yedid.hoshen@mail.huji.ac.il) \
Joseph Keshet (joseph.keshet@cs.biu.ac.il)             

DSegKNN, is an unsupervised kNN-based approach for word segmentation in speech utterances. This method relies on self-supervised pre-trained speech representations, and compares each audio segment of a given utterance to its K nearest neighbors within the training set. 


The paper can be found [here](https://arxiv.org/pdf/2204.13094.pdf). 


------


## Installation instructions

- Python 3.8+ 

- Pytorch 1.10.0

- torchaudio 0.10.0

- numpy

- scipy

- faiss

- soundfile

- Download the code:
    ```
    git clone https://github.com/MLSpeech/DSegKNN.git
    ```


## How to use

In this example, we will demonstrate how to run DSegKNN on the [Buckeye]() dataset. 

- We use the same experimental setup as in Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation (INTERSPEECH 2020)([Paper](https://arxiv.org/pdf/2007.13465.pdf), [Code](https://github.com/felixkreuk/UnsupSeg), script by F[elix Kreuk](https://felixkreuk.github.io/)):

	 - split long wavs into smaller chunks (cut during silences)
	 - leave 0.2 seconds of silence in the beginning and end
	 - there are no non-speech utterances
 
	Run the script as follows:

	```python buckeye_preprocess.py --spkr --source buckeye/speech/ --target datasets/buckeye_split/ --min_phonemes 20 --max_phonemes 50```

	This should create `train, val` and `test` folders in your chosen target directory `buckeye_split`. Each folder cut `.wav` files, with corresponding `.word` and '.phn' files containig the start and end times of words/phonemes within the `.wav` file.


- Run ```run_segmenter.py``` with the following options:

	```
	python knn_segmenter.py --win [number of frames to concatenate]
				 --train_n [number of training examples to use] 
				 --eval_n [number of evaluation examples to use]
				 --layer [index of output layer of embedding architecture]
				 --knn [number of nearest neighbors to compare to]
				 --arc [architecture name: BASE || LARGE || LARGE_LV60K || XLSR53 || HUBERT_BASE || HUBERT_LARGE || HUBERT_XLARGE]
				 --width [parameter for scipy.signal's find_peaks]
				 --distance [parameter for scipy.signal's find_peaks]
				 --prominence [parameter for scipy.signal's find_peaks]
				 --train_dir [path to training directory]
				 --val_dir [path to validation directory]
							
							
							
							
	```
	
	For example:

	```
	python knn_segmenter.py --win 10
				 --train_n 200
				 --eval_n -1
				 --layer 13
				 --knn 20
				 --arc HUBERT_LARGE
				 --width 2
				 --distance 4
				 --prominence 4
				 --train_dir datasets/buckeye_split/train/
				 --val_dir datasets/buckeye_split/val/

	```

	Should result with:

	```
	Final result: 31.015404643089606 32.232243517474635 31.612118531623173 3.923337091319068 40.71275576844716
	```

	(There could be some slight differences in results because 200 randomly drawn training examples are used).

	- For comparison, the evaluation script ```eval_segmentation.py``` used here is by [Herman Kamper](https://github.com/kamperh/vqwordseg/blob/main/eval_segmentation.py).

