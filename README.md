# Seedlings Classification

Training a CNN to predict plant seedling species (Kaggle playground competition).

## Model Tuning Log

- Using image size of 64, batch size of 32, and 40 epochs, the model gets ~99% training accuracy, and 74% dev accuracy.
- Trying 20% dropout, we get 95.8% training accuracy, and 75.8% dev accuracy.
- So far, using 40% dropout, a max filter size of 64 (larger model), and training for 200 epochs gets the best performance (train accuracy = 93.42%, dev accuracy = 85.49%).
- Adding data augmentation to the training dataset has increased the bias again.
- Increasing initial_filters to 32 and max_filters to 128 reduced the bias and the variance (98.5% train accuracy, 86.41% dev accuracy). This increased the overall capacity of the network to 671k parameters.
- Using a pre-trained EfficientNetB3 and fine-tuning it for 40 epochs with some L2 regularization and a new final dense hidden layer on the task gives ~94% dev accuracy, and 99.99% train accuracy.