# genre-classification-GTZAN-CNN

This project uses the GTZAN dataset, which is a dataset that contains 100 audio files each of 10 genres. It is a dataset used for benchmarking genre classification machine learning algorithms. For training purposes, each audio file has been divided into 5 segments for each of which 20 Mel Frequency Cepstral Coefficients have been extracted. The number 20 was arrived through a process of approximating how many coefficients led to the maximum efficiency.

This is then fed into a keras convolutional neural network which has 3 2D-convolutional layers, one dense layer and an output layer of 10 nodes. Upon training an accuracy of 95% was reached on the training set, along with 84% each on the validation and testing set. Through some random trials, a tendency to identify classic rock songs as country music has been identified. 


#### File Explainer:
1. GTZAN-dataset-music-genre-classification/: The dataset
2. mfcc_model.ipynb: The colab notebook in which the dataset was preprocessed and the neural network was made.
3. preprocessing.py: The file containing backend functions
4. genres.py: The genres enum class file.
5. app.py: The Flask backend app
6. Templates/ : The folder containing the HTML templates
7. Upload_Folder/ : The folder where audio files are uploaded by the user. 
