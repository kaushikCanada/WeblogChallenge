# %%
""" Import the required libraries. """
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lit, count, split, avg
import os
import sys
import findspark

findspark.init()

# %%
""" Create sparksession object """
spark = SparkSession.builder.appName("Paytm_client").master("local[*]").getOrCreate()

# %%
""" session window time(minutes) = 15  """
minute_limit_session = 15

# %%
""" Schema of the input file """
schema = StructType(
    [
        StructField("timestamp", StringType(), True),
        StructField("elb", StringType(), True),
        StructField("client_ip_port", StringType(), True),
        StructField("backend_ip_port", StringType(), True),
        StructField("request_processing_time", StringType(), True),
        StructField("backend_processing_time", FloatType(), True),
        StructField("response_processing_time", FloatType(), True),
        StructField("elb_status_code", IntegerType(), True),
        StructField("backend_status_code", IntegerType(), True),
        StructField("received_bytes", IntegerType(), True),
        StructField("sent_bytes", IntegerType(), True),
        StructField("request", StringType(), True),
        StructField("user_agent", StringType(), True),
        StructField("ssl_cipher", StringType(), True),
        StructField("ssl_protocol", StringType(), True),
    ]
)

# %%
""" Read input file stored in local. use the schema above. """
elb_logs_df = (
    spark.read.schema(schema)
    .option("delimiter", " ")
    .csv(
        """data\\2015_07_22_mktplace_shop_web_log_sample.log.gz"""
    )
)

# %%
""" Print the schema to see if it was made right. """
elb_logs_df.printSchema()

# %%
""" Here we are filtering some log entries where there is no backend ip port.
Also convert the string time to actual timestamp.
Select only the required entries. """
elb_logs_filtered_df = elb_logs_df.filter("backend_ip_port != '-'").select(
    col("timestamp").cast("timestamp").alias("timestamp"),
    "client_ip_port",
    "request",
    "user_agent",
)

# %%
""" This window captures the essence that ip and user agent combined provides unique visitor info.
Obviously order by timestamp. """
window_def_1 = Window.partitionBy("client_ip_port", "user_agent").orderBy("timestamp")

# %%
# elb_logs_sessionized_df.filter("session_unique_flag==1").show()

# %%
""" This is where the sessionization happens. The logs are sessionized using fifteen minute time windows.
Session id is constructed by combining ip,user agent, and the point at which it creates a session. 
Finally we repartition the data to help speed up processing. """

elb_logs_sessionized_df = (
    elb_logs_filtered_df.withColumn("timestamp", col("timestamp").cast("timestamp"))
    .withColumn("start_time", lag("timestamp", 1).over(window_def_1))
    .withColumn("start_time", coalesce("start_time", "timestamp"))
    .withColumn(
        "duration_of_request",
        unix_timestamp(("timestamp")) - unix_timestamp(col("start_time")),
    )
    .withColumn(
        "session_unique_flag",
        when(col("duration_of_request") > (minute_limit_session * 60), 1).otherwise(0),
    )
    .withColumn("session_point", sum("session_unique_flag").over(window_def_1))
    .withColumn(
        "session_id",
        concat(
            col("client_ip_port"),
            lit("_"),
            col("user_agent"),
            lit("_"),
            col("session_point"),
        ),
    )
    .withColumn(
        "duration_of_request",
        when(col("session_unique_flag") == 1, 0).otherwise(col("duration_of_request")),
    )
    .repartition("client_ip_port")
    .orderBy("timestamp")
    .cache()
)

# %%
# elb_logs_sessionized_df.filter("session_unique_flag==1 and duration_of_request<1000").select("session_id","duration_of_request").show(
#     100, truncate=False
# )

# %%
""" The average session time is Total Session Duration / Total Number of Sessions """
elb_logs_avg_session_df = (
    elb_logs_sessionized_df.groupBy("session_id")
    .agg(sum("duration_of_request").alias("duration_of_request"))
    .select(avg("duration_of_request").alias("avg_duration_of_request"))
)

# %%
# elb_logs_avg_session_df.show()
print(
    "The average session time is : "
    + str(elb_logs_avg_session_df.first().avg_duration_of_request)
    + " seconds"
)

# %%
""" The unique urls visits per session can be obtained by counting after dropping duplicates. """
elb_logs_unique_urls_df = (
    elb_logs_sessionized_df.withColumn("just_url", split(col("request"), " ")[1])
    .dropDuplicates(subset=["session_id", "request"])
    .groupBy("session_id")
    .count()
)

# %%
# elb_logs_unique_urls_df.show()

# %%
""" The most engaged users are the ones with maximum duration times of their session """
elb_logs_max_duration_sessions_df = (
    elb_logs_sessionized_df.groupBy("client_ip_port", "session_id")
    .agg(sum("duration_of_request").alias("duration_of_request"))
    .orderBy(col("duration_of_request").desc())
)


# %%
# elb_logs_max_duration_sessions_df.filter("session_id like '%_4' or session_id like '%_5'").select("session_id","duration_of_request").orderBy(col("session_id").desc()).show(100,truncate=False)

# %%
""" Save the results in local
 """
elb_logs_sessionized_df.repartition(1).write.option("header", "true").csv("question1")
elb_logs_avg_session_df.repartition(1).write.option("header", "true").csv("question2")
elb_logs_unique_urls_df.repartition(1).write.option("header", "true").csv("question3")
elb_logs_max_duration_sessions_df.repartition(1).write.option("header", "true").csv(
    "question4"
)

# %%
