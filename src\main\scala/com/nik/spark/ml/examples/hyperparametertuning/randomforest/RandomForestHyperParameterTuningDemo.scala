package com.nik.spark.ml.examples.hyperparametertuning.randomforest

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import scala.Range
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator,BinaryClassificationEvaluator}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
object RandomForestHyperParameterTuningDemo {
	def main(args: Array[String]) {
		// Optional: Use the following code below to set the Error reporting
		import org.apache.log4j._
		//Logger.getLogger("org").setLevel(Level.ERROR)

		// Spark Session
		val spark = SparkSession.builder().master("local[*]").getOrCreate()

		// Use Spark to read in the Titanic csv file.
		val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("adult-training.csv")

		// Print the Schema of the DataFrame
		data.printSchema()

		///////////////////////
		/// Display Data /////
		/////////////////////
		val colnames = data.columns
		val firstrow = data.head(1)(0)
		println("\n")
		println("Example Data Row")
		for (ind <- Range(1, colnames.length)) {
			println(colnames(ind))
			println(firstrow(ind))
			println("\n")
		}

		////////////////////////////////////////////////////
		//// Setting Up DataFrame for Machine Learning ////
		//////////////////////////////////////////////////
		import spark.implicits._
		// Grab only the columns we want
		val dataSetAll = data.select($"income", $"workclass", $"fnlwgt", $"education", $"education-num", $"marital-status", $"occupation", $"relationship", $"race", $"sex", $"capital-gain", $"capital-loss", $"hours-per-week", $"native-country")
		val dataSet = dataSetAll.na.drop()

		// A few things we need to do before Spark can accept the data!
		// Convert categorical columns into a binary vector using one hot encoder
		// We need to deal with the Categorical columns

		// Import VectorAssembler and Vectors
		import org.apache.spark.ml.feature.{ VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder }
		import org.apache.spark.ml.linalg.Vectors

		// Deal with Categorical Columns
		// Transform string type columns to string indexer 
		val workclassIndexer = new StringIndexer().setInputCol("workclass").setOutputCol("workclassIndex")
		val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
		val maritalStatusIndexer = new StringIndexer().setInputCol("marital-status").setOutputCol("maritalStatusIndex")
		val occupationIndexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIndex")
		val relationshipIndexer = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIndex")
		val raceIndexer = new StringIndexer().setInputCol("race").setOutputCol("raceIndex")
		val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
		val nativeCountryIndexer = new StringIndexer().setInputCol("native-country").setOutputCol("nativeCountryIndex")
		val incomeIndexer = new StringIndexer().setInputCol("income").setOutputCol("incomeIndex")

		// Transform string type columns to string indexer 
		val workclassEncoder = new OneHotEncoder().setInputCol("workclassIndex").setOutputCol("workclassVec")
		val educationEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationVec")
		val maritalStatusEncoder = new OneHotEncoder().setInputCol("maritalStatusIndex").setOutputCol("maritalVec")
		val occupationEncoder = new OneHotEncoder().setInputCol("occupationIndex").setOutputCol("occupationVec")
		val relationshipEncoder = new OneHotEncoder().setInputCol("relationshipIndex").setOutputCol("relationshipVec")
		val raceEncoder = new OneHotEncoder().setInputCol("raceIndex").setOutputCol("raceVec")
		val sexEncoder = new OneHotEncoder().setInputCol("sexIndex").setOutputCol("sexVec")
		val nativeCountryEncoder = new OneHotEncoder().setInputCol("nativeCountryIndex").setOutputCol("nativeCountryVec")
		val incomeEncoder = new StringIndexer().setInputCol("incomeIndex").setOutputCol("label")

		// Assemble everything together to be ("label","features") format
		val assembler = (new VectorAssembler()
				.setInputCols(Array("workclassVec", "fnlwgt", "educationVec", "education-num", "maritalVec", "occupationVec", "relationshipVec", "raceVec", "sexVec", "capital-gain", "capital-loss", "hours-per-week", "nativeCountryVec"))
				.setOutputCol("features"))

		////////////////////////////
		/// Split the Data ////////
		//////////////////////////
		val Array(training, test) = dataSet.randomSplit(Array(0.7, 0.3), seed = 12345)

		///////////////////////////////
		// Set Up the Pipeline ///////
		/////////////////////////////
		import org.apache.spark.ml.Pipeline

		val randomForest = new RandomForestClassifier()
		.setFeaturesCol("features")
		.setLabelCol("label")

		//create a grid of parameters
		val paramGrid = new ParamGridBuilder()
	//	.addGrid(randomForest.maxBins, Array(25, 28, 31))
	//	.addGrid(randomForest.maxDepth, Array(4, 6, 8))
	//	.addGrid(randomForest.impurity, Array("entropy", "gini"))
		.build()

		//create MulticlassClassificationEvaluator evaluator and use  area under the precision-recall curve as a metric
		val evaluator = new MulticlassClassificationEvaluator()
		.setLabelCol("label")
		.setPredictionCol("prediction")
	//	.setMetricName("areaUnderPR")
	  
		//create instance of pipeline
		val pipeline = new Pipeline().setStages(Array(workclassIndexer, educationIndexer, maritalStatusIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, nativeCountryIndexer, incomeIndexer, workclassEncoder, educationEncoder, maritalStatusEncoder, occupationEncoder, relationshipEncoder, raceEncoder, sexEncoder, nativeCountryEncoder, incomeEncoder, assembler,randomForest))

		//select best model based on TrainValidationSplit technique
		val trainValidationSplit = new TrainValidationSplit()
		.setEstimator(pipeline)
		.setEvaluator(evaluator)
		.setEstimatorParamMaps(paramGrid)
		.setTrainRatio(0.8)

		// Fit the pipeline to training documents.
		val model = trainValidationSplit.fit(training)

		// Get Results on Test Set
		val results = model.transform(test)
		model.transform(test)
		.select("features", "label", "prediction")
		.show()

		////////////////////////////////////
		//// MODEL EVALUATION /////////////
		//////////////////////////////////
		println("schema")
		println(results.select($"label").distinct().foreach { x => println(x) })

		// For Metrics and Evaluation
		import org.apache.spark.mllib.evaluation.MulticlassMetrics

		// Need to convert to RDD to use this
		val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

		// Instantiate metrics object
		val metrics = new MulticlassMetrics(predictionAndLabels)

		// Confusion matrix
		println("Confusion matrix:")
		println(metrics.confusionMatrix)
		println(metrics.accuracy)

		val evaluator1 = new MulticlassClassificationEvaluator()
		.setLabelCol("label")
		.setPredictionCol("prediction")
		.setMetricName("accuracy")
		val accuracy = evaluator.evaluate(results)
		println("Test Error = " + (1.0 - accuracy))
	}
}
