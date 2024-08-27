# To install kafka python: install kafka-python

# To start or stop Zookeeper and Kafka Services in Ubuntu after you have installed it
# sudo systemctl start zookeeper
# sudo systemctl start kafka
# sudo systemctl status zookeeper
# sudo systemctl status kafka
# sudo systemctl stop zookeeper
# sudo systemctl stop kafka
# kafka installation folder in Ubuntu: /usr/local/kafka/bin

# To start or stop Zookeeper and Kafka Services in Windows
# use powershell and navigate to the folder where kafka in installed or start cmd from that folder
# Command to start zookeeper services: .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
# Command to start kafka services: .\bin\windows\kafka-server-start.bat .\config\server.properties
# Command to stop kafka services: .\bin\windows\kafka-server-stop.bat or Ctrl + C on cmd/Powershell
# Command to stop zookeeper services: .\bin\windows\zookeeper-server-stop.bat or Ctrl + C on cmd/Powershell

# Note: Always start Zookeeper services before starting kafka services and always stop kafka services first before stopping zookeeper services

# imports
from kafka import KafkaProducer
from datetime import datetime
import logging
import pandas as pd
import os

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

logger = logging.getLogger("kafka-logger")

def produce_data(topic, data):
    """
    The function uses KafkaProducer function to stream data as topics
    Argumants-
        topic: topic name of the payload
        data: the payload assigned to topic 
    """
    producer = KafkaProducer(bootstrap_servers = 'localhost:9092')
    producer.send(topic, data.encode('utf-8'))
    producer.flush()

SOURCE_PATH = "C:/path/to/file"
DATA_FILE_NAME = os.path.join(SOURCE_PATH, 'BankChurners.csv')
# read csv file
df = pd.read_csv(DATA_FILE_NAME)

# drop features which has Naive_Bayes in them, these columns were not useful for this usecase
drop_columns = [col for col in df.columns if 'Naive_Bayes' in col]
df.drop(drop_columns, axis = "columns", inplace = True)

# number of rows to read from dataframe
n_rows = 10000
temp_df = df.head(n_rows)

start = datetime.now()
# iterate over rows from the dataframe
for index, data in temp_df.iterrows():
    # each row from the dataframe gets streamed as a topic
    produce_data(f"topic_{index}", data.to_json())
    logger.info(f"-- Completed topic_{index}")
logger.info("Total time taken to process all the batches : {}".format(datetime.now() - start))


