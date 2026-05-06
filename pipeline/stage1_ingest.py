from pyspark.sql import SparkSession 

spark = SparkSession.builder \
    .appName("sports_modelling_stage") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv("data/ENG/*.csv", header = True, inferSchema = True)
df.printSchema()
df.show(5)