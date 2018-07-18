package com.nik.spark.ml.examples.recommendation

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import scala.Range
import org.apache.spark.ml.feature.{ VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder }
import org.apache.spark.ml.Pipeline

object ProductRecommender {
	def main(args: Array[String]) {
		// Optional: Use the following code below to set the Error reporting
		import org.apache.log4j._
		Logger.getLogger("org").setLevel(Level.ERROR)

		// Spark Session
		val spark = SparkSession.builder().master("local[*]").getOrCreate()
		import spark.implicits._

		//read json file of digitial music products reviewed on AMazon
		val ratings = spark.read.json("Digital_Music_5.json")

		//Display first record and schema
		ratings.head()
		ratings.printSchema()

		//////// We will be using Collaborative Filtering technique for a recommendation system
		///////  In order to perform collaborative filtering ALS - Alternating Least Squares

		//ALS technique requires user_id, product_id categorical columns in numerical format
		//In this dataset reviewerID is user_id and asin is product_id
		//These columns are categorical and in String format
		//In order to use these columns in ALS, transform them to numerical using StringIndxer
		val featureCol = ratings.select("reviewerID", "asin").columns
		val indexers = featureCol.map { colName =>
		new StringIndexer().setInputCol(colName).setOutputCol(colName + "Index")
		}
		val pipelineIndexer = new Pipeline()
		.setStages(indexers)
		val dataIndexed = pipelineIndexer.fit(ratings).transform(ratings)

		//print schema of data set with transformed columns
		dataIndexed.printSchema()

		//split training and test data
		val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

		// Build the recommendation model using ALS on the training data
		val als = new ALS()
		.setMaxIter(5)
		.setRegParam(0.01)
		.setUserCol("reviewerIDIndex")
		.setItemCol("asinIndex")
		.setRatingCol("overall")

		val model = als.fit(dataIndexed)

		// Evaluate the model by computing the RMSE on the test data
		// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
		model.setColdStartStrategy("drop")

		// Evaluate the model by computing the average error from real rating
		val predictions = model.transform(test)
		predictions.printSchema()

		val evaluator = new RegressionEvaluator()
		.setMetricName("rmse")
		.setLabelCol("overall")
		.setPredictionCol("prediction")
		
		val rmse = evaluator.evaluate(predictions)
		println(s"Root-mean-square error = $rmse")

		// Generate top 10 movie recommendations for each user
		val userRecs = model.recommendForAllUsers(10)
		// Generate top 10 user recommendations for each movie
		val movieRecs = model.recommendForAllItems(10)
	}
}