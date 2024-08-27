# install pyspark using pip install pyspark==3.4.3
# To submit spark job
# spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 --driver-class-path postgresql-42.7.4.jar --jars postgresql-42.7.4.jar

# imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

### Log Configuration
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
)

logger = logging.getLogger("spark-logger")

# defined schema for consuming the data from kafka stream
json_schema = StructType([
    StructField("CLIENTNUM", StringType()),
    StructField("Attrition_Flag", StringType()),
    StructField("Customer_Age", IntegerType()),
    StructField("Gender", StringType()),
    StructField("Dependent_count", IntegerType()),
    StructField("Education_Level", StringType()),
    StructField("Marital_Status", StringType()),
    StructField("Income_Category", StringType()),
    StructField("Card_Category", StringType()),
    StructField("Months_on_book", IntegerType()),
    StructField("Total_Relationship_Count", IntegerType()),
    StructField("Months_Inactive_12_mon", IntegerType()),
    StructField("Contacts_Count_12_mon", IntegerType()),
    StructField("Credit_Limit", FloatType()),
    StructField("Total_Revolving_Bal", FloatType()),
    StructField("Avg_Open_To_Buy", FloatType()),
    StructField("Total_Amt_Chng_Q4_Q1", FloatType()),
    StructField("Total_Trans_Amt", FloatType()),
    StructField("Total_Trans_Ct", FloatType()),
    StructField("Total_Ct_Chng_Q4_Q1", FloatType()),
    StructField("Avg_Utilization_Ratio", FloatType()),
])

def consume(topic):
    """
    The function takes in the topic name and retrieves its data from kafka
    """
    # create a new spark session
    spark = SparkSession.builder \
    .appName('KafkaConsumer') \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2") \
    .config("spark.jars.packages", "path/to/file/postgresql-42.7.4.jar") \
    .getOrCreate()

    # create a new spark dataframe by reading the data from the topic
    df = (
        spark.read.format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", topic) \
        .load() \
        .selectExpr("CAST(value AS STRING) as json_data") \
        .select(from_json(col("json_data"), json_schema).alias("data")) \
        .select("data.*")
    )
    
    # Add a new timestamp column which determines the point at which the row was fetched.
    df = df.withColumn("TimeStamp", current_timestamp())
    
    #print(df.show())
    
    # write this data to postgres database
    df.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/<DatabaseName>") \
    .option("driver", "org.postgresql.Driver") \
    .option("dbtable", '<SchemaName>."<TableName>"') \
    .option("user", "<postgres_username>") \
    .option("password", "<postgres_password>") \
    .mode("append") \
    .save()
    
    

N_BATCHES = 10000
for n in range(N_BATCHES):
    consume(f"topic_{n}")
    logger.info(f"-- Completed topic_{n}")