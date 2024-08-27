# In this file we are going to read consumed data from kafka which is stored in postgresql and build a small data processing pipeline
import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from configparser import ConfigParser
import os

def load_config(filename = "postgresDB.ini", section = "postgresql"):
    """
    Loading postgres configuration from config file
    """
    parser = ConfigParser()
    parser.read(filename)
    
    # get section, default to postgresql
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(f"[{section}] not found in [{filename}]")
    
    return config

def connect(config):
    """
    Connect to postgresql using the connection params
    """
    try:
        with psycopg2.connect(**config) as conn:
            #print(conn)
            print("POSTGRES_CONNECTION_LOG: Connected to PostgreSQL server...")
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
        
def execute_query(sql_statement, conn):
    cursor = conn.cursor()
    cursor.execute(sql_statement)
    rows = cursor.fetchall()
    cursor.close()
    return rows

postgres_config = load_config(filename = "postgresDB.ini")

postgres_conn = connect(postgres_config)

# execute the sql query to read all the records from RawData table
select_query = """
SELECT * FROM public."RawData";
"""
db_rows = execute_query(select_query, postgres_conn)
postgres_conn.close()

column_names = ['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
       'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

df = pd.DataFrame(db_rows, columns = column_names)

# convert labels from below columns to appropriate numerical index
df["Attrition_Flag"] = df["Attrition_Flag"].apply(lambda x: 1 if x == "Existing Customer" else 0)
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "F" else 0)

uid_column = ["CLIENTNUM"]
# filter all numeric columns
df_datatypes = df.dtypes[df.dtypes != object]
numeric_columns = df_datatypes.index.to_list()

# iterate over all the numeric columns and convert them into range of [0, 1] if the max value of that column is greater tham 1
for col in numeric_columns:
    if df[col].max() > 1:
        mm_scaler = MinMaxScaler()
        df[col] = mm_scaler.fit_transform(df[col].values.reshape(-1, 1))

final_columns = uid_column + numeric_columns

# store processed data in a parquet file
if not os.path.exists("processedData.parquet"):
    df[final_columns].to_parquet("processedData.parquet")