# BC-Wisconsin-Dataset
### **References** : [UCI Machine Learning Repository (Breast Cancer Wisconsin Dataset)](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)

## **Comprehensive Report on Breast Cancer Diagnosis Prediction Using Machine Learning**
### **1. Introduction**

Breast cancer is one of the leading causes of cancer-related deaths worldwide. Early detection plays a crucial role in improving survival rates. In this report, we explore the use of machine learning algorithms to predict whether a breast scan can be diagnosed as benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Dataset. The dataset consists of features extracted from digitized breast cancer scans, and the objective is to classify these scans into two categories: **`benign`** (non-cancerous) and **`malignant`** (cancerous).

To achieve this, five machine learning algorithms were tested:
- Logistic Regression
- Neural Network
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The goal is to evaluate and rank the performance of these models based on their accuracy and other evaluation metrics, focusing on the impact of preprocessing steps such as data encoding, sampling techniques, and normalization.

### **2. Dataset Overview**

The dataset used in this study, the Breast Cancer Wisconsin (Diagnostic) dataset, contains features extracted from breast cancer biopsies, including measurements like radius, texture, smoothness, compactness, concavity, and symmetry. These features are used to predict whether a tumor is benign or malignant.

The dataset consists of:
- Data Features: Various numeric measurements describing cell nuclei (e.g., radius, texture, smoothness).
- Target Variable: The diagnosis (Benign or Malignant).

### **3. Data Preprocessing**

Before training the machine learning models, several preprocessing steps were carried out to optimize the data for analysis:
- Encoding of Target Variable: The target variable, "Diagnosis," was encoded to convert categorical values (Benign, Malignant) into numeric values **`(0, 1)`** for compatibility with machine learning algorithms.
- Sampling Techniques: To address any class imbalance, random undersampling and random oversampling techniques were applied to balance the dataset and ensure fair representation of both classes (benign and malignant tumors).
- Feature Scaling: MinMax scaling was applied to normalize the feature data. This scaling method transformed the data into a range between 0 and 1, ensuring that all features had equal importance during model training, particularly for models sensitive to the scale of input features.

### **4. Machine Learning Algorithms**

The following algorithms were tested:
- Logistic Regression: Logistic Regression models the probability of a binary outcome (benign or malignant) using a linear decision boundary. It was expected to perform well with linear relationships between features.
- Neural Network: A neural network was used to model complex, non-linear relationships between features. The network was trained using backpropagation and adjusted for optimal performance.
- Random Forest: A Random Forest algorithm, an ensemble method based on decision trees, was used to model complex, non-linear relationships and handle interactions between features effectively. Random Forest is not sensitive to feature scaling, making it robust for this dataset.
- Support Vector Machine (SVM): The Support Vector Machine algorithm aims to find a hyperplane that best separates the two classes. Since SVM is sensitive to feature scaling, the data was normalized to improve model performance.
- XGBoost: XGBoost is an advanced implementation of gradient boosting that uses decision trees to model non-linear relationships. It was applied to capture complex interactions between features.

### **5. Model Evaluation**

To assess the performance of the models, we used the following metrics:
- Accuracy: The proportion of correctly classified instances.
- Cross-Validation Scores: To ensure the model is generalizing well and not overfitting, we used k-fold cross-validation to estimate model performance on unseen data.
- Confusion Matrix: To evaluate the performance of the models in terms of false positives, false negatives, true positives, and true negatives.
- Classification Report: To compute precision, recall, and F1-score for each model.

### **6. Results**
**Model Accuracy:**
After preprocessing the data, the following accuracy results were observed:
- Logistic Regression: 98%
- Neural Network: 98%
- Support Vector Machine (SVM): 98%
- Gradient Boosting (XGBoost): 97%
- Random Forest: 97%

### **7. Observations**
- Logistic Regression: Achieved an impressive accuracy of 98%, improved from a previous accuracy of 92%. This indicates that the relationships in the data are somewhat linear, and normalization improved its performance.
- Neural Network & SVM: Both models, which initially performed poorly (53% and 49%, respectively), achieved 98% accuracy after hyperparameter tuning and normalization. This demonstrates the importance of preprocessing for these models, which are sensitive to the scale of features.
- Random Forest & XGBoost: These tree-based models performed well, achieving 97% accuracy. Their performance suggests they were effective at capturing complex, non-linear relationships in the data, even without normalization.

### **8. Discussion**
- Feature Correlation: The feature correlation analysis showed that there were no clear linear relationships between individual features and the target variable. This supports the use of more advanced models like Random Forest and XGBoost, which can capture complex relationships that linear models like Logistic Regression might miss.
- Importance of Normalization: All models, particularly Neural Networks and SVM, showed significant improvement after data normalization. This highlights the importance of preprocessing steps in machine learning pipelines, particularly when using models that are sensitive to the scale of input data.
- Model Performance: While Logistic Regression performed well due to linear relationships in the data, more complex models like Random Forest and XGBoost may offer more robust performance when the relationships between features are intricate and non-linear. The improvements seen in Neural Networks and SVM after preprocessing indicate that, with the right tuning, these models can perform on par with tree-based models.

### **9. Conclusion**

The machine learning models tested—Logistic Regression, Neural Network, Random Forest, SVM, and XGBoost—all showed strong predictive performance in diagnosing breast cancer. The results indicate that while Logistic Regression is effective when linear relationships are present, more complex models like Random Forest and XGBoost offer superior performance when non-linear interactions are involved.

Normalization played a crucial role in enhancing the performance of models sensitive to feature scaling, such as Neural Networks and SVM. Future work can explore further hyperparameter tuning, additional feature engineering, and the use of more advanced models to further optimize the predictive accuracy for breast cancer diagnosis.

### **10. Recommendations for Future Work**
- Hyperparameter Tuning: Further optimization of model parameters, especially for Neural Networks and SVM, could lead to even better performance.
- Feature Engineering: Additional features, such as statistical or domain-specific attributes, could improve model performance.
- Ensemble Methods: Combining multiple models (e.g., a Random Forest and XGBoost ensemble) might lead to a more robust classifier.
