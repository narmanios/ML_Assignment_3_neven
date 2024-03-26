# Assignment 3:  Support Vector Machines classifier (SVC)

This notebook provides a comprehensive walkthrough of Machine Learning Assignment 3, which is centered around building and tuning a Support Vector Machines (SVM) classifier for sentiment analysis of movie reviews. The assignment involves experimenting with various regularization parameters to optimize the model's performance.

## Dependencies

- Python 3.x
- NumPy
- pandas
- Matplotlib
- scikit-learn

## Feature Extraction and Preprocessing

The raw movie review data is processed to extract features suitable for machine learning models. This includes:

- Vectorizing the text using `CountVectorizer` to create a bag-of-words representation.
- Applying `TfidfTransformer` to convert the bag-of-words representation into TF-IDF features.
- Adding quantitative features such as word count and punctuation count.
- Standardizing the features using `StandardScaler`.

## Model Training and Evaluation

The `LinearSVC` model from scikit-learn is trained with various values of the regularization parameter `C`. The performance of each model is evaluated using a custom `BinaryClassificationPerformance` class, which calculates measures such as accuracy, precision, and recall.

## Results

The performance of each model is visualized using ROC plots for both the training and test sets. The analysis helps in understanding the impact of the regularization parameter on the model's ability to generalize and predict sentiment accurately.

## Usage

To replicate the analysis, run the provided Jupyter notebook. Make sure to have all the required dependencies installed and the dataset properly formatted.

## Author

Neven Armanios
