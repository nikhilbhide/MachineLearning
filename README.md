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

### How to Run?
#### With Scala-Eclipse
Import project into IDE. Create a new Run Configuration and execute respective scala object files


#### With Spark-Submit
Create fat-jar using maven package and run spark-submit.
./spark-submit \
--class <qualified_class_name> \
<jar_location> \

