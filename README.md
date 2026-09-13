Building, Breaking and Fixing a Neural Network

A feedforward neural network pipeline on Fashion-MNIST, built end-to-end in seven parts: manual backpropagation, activation and loss-function studies, optimiser comparison, deliberate overfitting, a regularisation study, and hyperparameter tuning with cross-validation.
Overview of the notebook
Part	What it covers
0	Data loading, normalisation, flattening, and an 80/20 stratified train/validation split (test set held out)
1	A two-layer MLP backpropagation implementation from scratch in NumPy, verified against PyTorch autograd via a gradient check
2	A baseline MLP trained with sigmoid, tanh, ReLU, and leaky ReLU, comparing vanishing gradients and dead-ReLU units
3	Cross-entropy vs. mean squared error for classification, plus a small regression MLP on the diabetes dataset
4	SGD, SGD with momentum, RMSProp, and Adam compared at a shared learning rate and again after per-optimiser tuning
5	A deliberately overfit model (2,000 training samples, four hidden layers of 512 units) to create a large generalisation gap
6	A regularisation study: L2, L1, dropout, batch normalisation, early stopping, data augmentation, and additional training data
7	Random search (12 configurations) scored by 5-fold cross-validation, followed by a single final evaluation on the held-out test set

Requirements
Python 3.10+
numpy, pandas, matplotlib, seaborn
scikit-learn
torch, torchvision

Dataset

Fashion-MNIST (Zalando Research), via Kaggle Datasets. The notebook expects the CSV files at:

/kaggle/input/datasets/zalando-research/fashionmnist/fashion-mnist_train.csv
/kaggle/input/datasets/zalando-research/fashionmnist/fashion-mnist_test.csv

If running outside Kaggle, download the dataset from the link above and update the Data_dir path in the first data-loading cell to point to your local copy.

Reproducing the results
Platform: Kaggle Notebooks with the GPU T4 x2 accelerator enabled (Settings → Accelerator). The notebook will also run on CPU, but Part 6 and Part 7 (which train dozens of models) will take considerably longer.
Attach the dataset: add the Fashion-MNIST dataset above to the notebook's inputs, or update Data_dir to a local path.
Run all cells from top to bottom (Run → Run All, or Restart & Run All). The notebook is fully self-contained and deterministic — a single random seed (SEED = 42) is set at the top and reused throughout (NumPy, Python's random, and PyTorch, including torch.backends.cudnn.deterministic = True), so a clean top-to-bottom run reproduces every number reported here exactly.
Runtime: the full notebook — including the Part 7 cross-validated random search (60 model trainings) — takes roughly 15–25 minutes on a T4 GPU.
Outputs: each part's cell outputs (metrics, tables, plots) render inline as the notebook executes; a consolidated summary cell at the end reprints the key reportable value from every part in one place.
