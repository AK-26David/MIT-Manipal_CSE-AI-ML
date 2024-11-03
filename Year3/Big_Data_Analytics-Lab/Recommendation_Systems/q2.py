from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecommendationModelEvaluation") \
    .getOrCreate()

# Load dataset (assuming data has columns: userId, itemId, rating)
data = spark.read.csv("/Users/arnavkarnik/Documents/MIT-Manipal_CSE-AI-ML/Year3/Big_Data_Analytics-Lab/Recommendation_Systems/random_ratings.csv", header=True, inferSchema=True)
data = data.select("userId", "itemId", "rating")

# Split data into training and testing sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# Train ALS model
als = ALS(userCol="userId", itemCol="itemId", ratingCol="rating", maxIter=10, regParam=0.1, rank=10, coldStartStrategy="drop")
model = als.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluation - Regression Metrics
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
print(f"Root-Mean-Squared Error (RMSE) = {rmse}")
print(f"Mean Absolute Error (MAE) = {mae}")

# Ranking Metrics (Precision@K, Recall@K)

# Generate top-K recommendations for each user in test data
K = 5
user_recs = model.recommendForAllUsers(K)

# Explode recommendations to get a flat table of (userId, itemId, rating)
user_recs_expanded = user_recs \
    .withColumn("rec_exp", explode("recommendations")) \
    .select("userId", col("rec_exp.itemId").alias("itemId"), col("rec_exp.rating").alias("predicted_rating"))

# Join expanded recommendations with test data to find hits
hits = user_recs_expanded.join(test_data, ["userId", "itemId"]) \
    .select("userId", "itemId", "rating")

# Calculate Precision@K and Recall@K for each user
precision_k = hits.groupBy("userId").count().withColumnRenamed("count", "relevant_recommended") \
    .join(test_data.groupBy("userId").count().withColumnRenamed("count", "actual_relevant"), "userId") \
    .withColumn("precision", col("relevant_recommended") / K) \
    .withColumn("recall", col("relevant_recommended") / col("actual_relevant"))

# Aggregate Precision@K and Recall@K over all users
avg_precision_k = precision_k.selectExpr("avg(precision) as avg_precision").collect()[0]["avg_precision"]
avg_recall_k = precision_k.selectExpr("avg(recall) as avg_recall").collect()[0]["avg_recall"]

print(f"Precision@{K} = {avg_precision_k}")
print(f"Recall@{K} = {avg_recall_k}")

# Stop Spark session
spark.stop()
