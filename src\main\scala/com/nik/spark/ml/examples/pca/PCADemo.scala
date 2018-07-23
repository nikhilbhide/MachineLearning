package com.nik.spark.examples.pca

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import scala.Range
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator,BinaryClassificationEvaluator}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.linalg.Vectors

object PCADemo {
	def main(args: Array[String]) {
		// Optional: Use the following code below to set the Error reporting
		import org.apache.log4j._
		//Logger.getLogger("org").setLevel(Level.ERROR)

		// Spark Session
		val spark = SparkSession.builder().master("local[*]").getOrCreate()

		// Use Spark to read in the Cancer data csv file.
		val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Cancer_Data")

		// Print the Schema of the DataFrame
		data.printSchema()

		//create an array of columns so that it can be used in assembler
		val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
				"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
				"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
				"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
				"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
				"worst concave points", "worst symmetry", "worst fractal dimension"))

		//necessary imports
		import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
		import org.apache.spark.ml.linalg.Vectors

		//create vector assembler with in order to wrap all dimensions under features column 
		val assembler = (new VectorAssembler()
				.setInputCols(colnames)
				.setOutputCol("features"))
		val featureData = assembler.transform(data)
		//print schema of transformed data set
		featureData.printSchema()

		// Use StandardScaler on the data
		// Create a new StandardScaler() object called scaler
		// Set the input to the features column and the ouput to a column called scaledFeatures
		val scaler = (new StandardScaler()
				.setInputCol("features")
				.setOutputCol("scaledFeatures")
				.setWithStd(true)
				.setWithMean(false))

		// Compute summary statistics by fitting the StandardScaler.
		// Basically create a new object called scalerModel by using scaler.fit()
		// on the output of the VectorAssembler
		val scalerModel = scaler.fit(featureData)

		// Normalize each feature to have unit standard deviation.
		// Use transform() off of this scalerModel object to create your scaledData
		val scaledData = scalerModel.transform(featureData)


		//create pca to reduce dimensions from 30 to 6 and apply pca on scaled data set
		val pca = (new PCA()
				.setInputCol("scaledFeatures")
				.setOutputCol("pcaFeatures")
				.setK(6)
				.fit(scaledData))
		val pcaDF = pca.transform(scaledData)

		// Transform using pca and check out the results
		// Check out the results
		// Show the new pcaFeatures
		val result = pcaDF.select("pcaFeatures")
		result.show()

		// Use .head() to confirm that your output column Array of pcaFeatures
		// only has 4 principal components
		result.head(1).foreach(x=>println(x))
	}
}
