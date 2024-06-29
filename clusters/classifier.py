import psycopg2
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from sklearn.mixture import GaussianMixture
import pandas as pd
import time

def fetch_data(host, port, database, user, password):
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

        cursor = connection.cursor()

        select_query = "SELECT * FROM marks"
        start_time = time.time()  # Start measuring time
        cursor.execute(select_query)
        records = cursor.fetchall()
        end_time = time.time() 
        cursor.close()
        df = pd.DataFrame(records, columns=[desc[0] for desc in cursor.description])
        return df

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL or fetching data:", error)
        return None

    finally:
        # Close communication with PostgreSQL
        if 'connection' in locals():
            if connection:
                connection.close()
                print("PostgreSQL connection is closed")

def preprocess_data(spark, spark_df):
    last_3_dates = spark_df.select('date').distinct().orderBy('date', ascending=False).limit(3).rdd.flatMap(lambda x: x).collect()

    filtered_data = spark_df.filter(spark_df.date.isin(last_3_dates))

    pivot_df = filtered_data.groupBy("id").pivot("date").agg({"marks": "first"})

    columns = ['id'] + [f'day{i+1}' for i in range(len(last_3_dates))]
    pivot_df = pivot_df.toDF(*columns)

    avg_marks = filtered_data.groupBy("date").agg(avg("marks").alias("avg_marks"))

    for i, date in enumerate(last_3_dates):
        avg_mark = avg_marks.filter(col("date") == date).select("avg_marks").collect()[0]["avg_marks"]
        pivot_df = pivot_df.fillna({f'day{i+1}': avg_mark})

    return pivot_df

def run_classifier(cluster_count=5):
    try:
        sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").setAppName("SparkSQL"))
        spark = SparkSession(sc)

        host = '192.168.8.175'
        port = '5000' 
        database = 'holomentor_db'
        user = 'postgres'
        password = 'postgres'

        data_df = fetch_data(host, port, database, user, password)
        if data_df is not None:
            spark_df = spark.createDataFrame(data_df)
        else:
            print("Data not fetched successfully")
            return
        
        preprocessed_df = preprocess_data(spark, spark_df)

        pandas_df = preprocessed_df.toPandas()

        X = pandas_df.drop("id", axis=1)

        gmm = GaussianMixture(n_components=cluster_count, random_state=0)
        pandas_df['cluster'] = gmm.fit_predict(X)

        pivot_df = spark.createDataFrame(pandas_df)

    except Exception as e:
        print("Error:", e)

    finally:
        if 'sc' in locals():
            sc.stop()
        if 'spark' in locals():
            spark.stop()
