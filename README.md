## Machine Learning Examples Using Spark

Apache Spark has inbuilt support of many machine learning algorithms.
In this repository, I have demonstrated the usage of Spark ML features by taking different case studies.
Note that we are going to use latest spark ML library that is org.apache.spark.ml, which works with data set.

Following are the case studies
1. LogRegDemo.scala - Classification using logistic regression
Case Study - US adult income prediction - 
https://www.kaggle.com/marksman/us-adult-income-salary-prediction
2. ProductRecommender.scala - Full fledged recommendation system using collaborative filtering (using Alternative Least Squaring)
Case study - Amazon music store product data
Digital_Music_5.json - 
http://jmcauley.ucsd.edu/data/amazon/
3. RandomForestDemo.scala - Classification using logistic regression
Case Study - US adult income prediction - 
https://www.kaggle.com/marksman/us-adult-income-salary-prediction
4. RandomForestHyperParameterTuningDemo.scala - Classification using Random Forest and model selection using hyperparametertuning.
Case Study - US adult income prediction - 
https://www.kaggle.com/marksman/us-adult-income-salary-prediction
5. PCADemo.scala - Dimension reduction using principle component analysis
Dimensionality reduction is the process of reducing the number of variables under consideration. It can be used to extract latent features from raw and noisy features or compress data while maintaining the structure. spark.mllib provides support for dimensionality reduction on the RowMatrix class.
Case Study - Breast Cancer Wisconsin (Diagnostic) Data Set
Predict whether the cancer is benign or malignant
https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

### How to Run?
#### With Scala-Eclipse
Import project into IDE. Create a new Run Configuration and execute respective scala object files


#### With Spark-Submit
Create fat-jar using maven package and run spark-submit.
./spark-submit \
--class <qualified_class_name> \
<jar_location> \

