from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecommendationModel") \
    .getOrCreate()

# Load dataset (assuming data has columns: userId, itemId, rating)
# Replace "your_data.csv" with the path to your dataset
data = spark.read.csv("/Users/arnavkarnik/Documents/MIT-Manipal_CSE-AI-ML/Year3/Big_Data_Analytics-Lab/Recommendation_Systems/random_ratings.csv", header=True, inferSchema=True)

# Prepare data (ensure columns have the right names for ALS)
data = data.select("userId", "itemId", "rating")

# Split data into training and testing sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# Initialize ALS model
als = ALS(
    userCol="userId",
    itemCol="itemId",
    ratingCol="rating",
    maxIter=10,         # Maximum number of iterations
    regParam=0.1,       # Regularization parameter
    rank=10,            # Latent factors in the model
    coldStartStrategy="drop"  # Drop NaN predictions
)

# Train ALS model on training data
model = als.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate model using RMSE (Root Mean Squared Error)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Generate recommendations for a subset of users
user_recs = model.recommendForAllUsers(numItems=5)  # Recommend 5 items per user
item_recs = model.recommendForAllItems(numUsers=5)  # Recommend 5 users per item

# Show example recommendations
user_recs.show(5, truncate=False)
item_recs.show(5, truncate=False)

# Stop Spark session
spark.stop()
