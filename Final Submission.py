# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Airline Flight Delay 
# MAGIC ### Team 20
# MAGIC ##### Sanjay Saravanan, Sean Campos, Devesh Khandelwal, and Karthik Rameshbabu
# MAGIC ##### 12/11/2020
# MAGIC ##### Course: Machine Learning at Scale (w261)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Introduction
# MAGIC 
# MAGIC Predicting Flight Delay has been an age old problem troubling travellers, airport administrators, and airline staff alike. After studying various pieces of literature in this space, our team has taken a stab at using **flight**, **weather**, and **airport** data to build machine learning models that will **predict whether a flight will be delayed, or not delayed,** based off a variety of features. We hope this report will shed light on our journey and our discoveries.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Context:
# MAGIC As we are all painfully aware, airline delays are all too common of an experience.  And as airports are one of the most resource intensive shared spaces in modern commerce, delays bring increased costs to crews, airport personnel, limited gate availability, fuel and maintenance cost as well as to the passengers.  The FAA reports that in 2019 flight delays cost the airlines more than $8 billion, passengers more than $18 billion, an additional $6 billion in lost demand and indirect costs.  Every minute of recovered delay time is worth at least $75.  While many of the event sequences that cause these delays are too complicated to fully prevent, predicting them and providing advanced notice to the multitude of interested parties would not only be a convenience, but a significant savings in wasted resources. Passengers would not have to spend as much time occupying expensive shared resources and airport personnel could take the necessary steps to allocate limited resources such as gates and scheduled equipment usage to not only save resources, but also mitigate the propagation of the delay downstream to additional flights, crews and passengers.
# MAGIC 
# MAGIC Our specific objective is to predict whether or not each flight will depart from the gate more than fifteen minutes after the scheduled departure time based on all of the information available two hours prior to departure.  These predictions are suitable for triggering both a customer centric notification system, such as push notifications through an air carrier’s app, as well as the airport’s internal resource coordination infrastructure. 
# MAGIC 
# MAGIC While every prediction algorithm strives for accuracy, we think it’s important to note that there may be a high cost to **false positives** associated with this system.  Passengers mistakenly believing that they have more time to arrive at the airport or misallocating airport resources for an incoming flight could be far more costly than the benefits of having a prediction system in the first place.  For this reason, **precision** will be our **most important metric** as it will force us to focus on reducing false positives.
# MAGIC 
# MAGIC In the previous publications that take a big data approach with similar datasets, we have found that Choi et. al. [1] achieved accuracy=0.80 with precision=0.71 and Patgiri et. al. [2] achieved accuracy=0.82 with precision=0.82.  In both papers, many machine learning algorithms were explored ranging from KNN to logistic regression and various tree based methods, yet the best performance was achieved with random forests.
# MAGIC 
# MAGIC 
# MAGIC > 1. S. Choi, Y. Kim, S. Briceno, and D. Mavris. Prediction of weather-induced airline delays based on machine learning algorithms.  In AIAA/IEEE Digital Avionics Systems Confer-ence - Proceedings, volume 2016-December, 2016. [https://ieeexplore.ieee.org/document/7777956]
# MAGIC > 2. R. Patgiri, S. Hussain, and A. Nongmeikapam. Empirical Study on Airline Delay Analysis and Prediction. In EAI Endorsed Transactions 2020. [https://arxiv.org/pdf/2002.10254.pdf]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC 
# MAGIC ### Raw Data
# MAGIC 
# MAGIC Here we pull in the following **raw** data tables and take a snapshot of the data size.
# MAGIC 
# MAGIC **Stations - 726.46 KB (.csv.gz)**
# MAGIC 
# MAGIC **Airlines - 1.52 GB (zip)**
# MAGIC 
# MAGIC | Year   | Size |
# MAGIC |-------|----------|
# MAGIC | 2015: | 276.03 MB  |
# MAGIC | 2016: | 277.71 MB  |
# MAGIC | 2017: | 284.77 MB  |
# MAGIC | 2018: | 351.83 MB  |
# MAGIC | 2019: | 364.75 MB  |
# MAGIC 
# MAGIC 
# MAGIC **Weather - 23.64 GB (parquet)**
# MAGIC 
# MAGIC | Year   | Size |
# MAGIC |-------|----------|
# MAGIC | 2015: | 4.49 GB  |
# MAGIC | 2016: | 4.69 GB  |
# MAGIC | 2017: | 4.72 GB  |
# MAGIC | 2018: | 4.73 GB  |
# MAGIC | 2019: | 4.87 GB  |
# MAGIC | MISS: | 144.74 MB  |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Imports
# MAGIC 
# MAGIC Included are all the imports we require for the project. Organized into the appropriate sections

# COMMAND ----------

#PySpark SQL
from pyspark.sql import SQLContext
from pyspark.sql import functions as f
from pyspark.sql import Window
import pyspark.sql.types as t # We can just do this and avoid large lsit of import types
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql.functions import concat, col, hour, minute, lpad, rpad, substring, year, month, dayofmonth, lit, to_timestamp, expr,split
from pyspark.sql.functions import isnan, when, count, col, to_timestamp, udf

#Utiliities
import pandas as pd
import numpy as np
import math
import us
# from haversine import haversine, Unit

#Date/Time
from datetime import datetime as dt, timedelta
from timezonefinder import TimezoneFinder
from pytz import timezone
import time
import pytz


#Plotting
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sn
import matplotlib.cm as cm, matplotlib.font_manager as fm

#Modelling
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, classification_report, confusion_matrix
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, StandardScaler, OneHotEncoder, SQLTransformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark



sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## EDA - Exploratory Data Analysis
# MAGIC 
# MAGIC In this section, we would like to showcase some of the key findings / descriptive statistics from our datasets. We have decided to include some markdown cells to highlight certain code chunks used in the EDA process. To ensure notebook clarity, we have chosen to only include the highlights of our EDA and links are included to separate notebooks that go into more detail for various data cleaning steps as well as additional exploration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stations Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### us_stations_all
# MAGIC In a search to find a way to join stations data with airlines and weather, our team looked to enrich the station tables with corresponding **IATA** airports codes. This would provide a deterministic foreign key across tables. 
# MAGIC 
# MAGIC This is when we came across this data source https://www.aviationweather.gov/docs/metar/stations.txt. Our took it as a task to scrape this data and parse it into a workable format, the work for this is linked in this [notebook].
# MAGIC 
# MAGIC From this we built a tabled called `us_stations_all` that we made available to all project teams via Databricks public tables
# MAGIC 
# MAGIC [notebook]: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275119900/command/4061768275119902

# COMMAND ----------

#us_stations_all sample
us_stations_all = spark.sql("select * from us_stations_all limit 5")
display(us_stations_all)

# COMMAND ----------

# MAGIC %md
# MAGIC The inital `us_stations_all` was a great start, however we intially ran into trouble when joining it with the airport table lead to dropped rows.  We used anti-joins to interrogate the missing rows and found that there were some missing and duplicate station values.
# MAGIC 
# MAGIC The second cleaning phase occured when joining the stations list with the weather table. Due to minor errors in the table, as well as some out of date information involving stations that had been replaced, certain overlaping time periods of data were causing joins to multiply the number of weathe observation rows. After more careful cleaning of the station data we were able to associate nearly 24 observations per day with most stations and quantify the amount of missing weather values that would need to be strategically filled in or accounted for.
# MAGIC 
# MAGIC In th graph below, we can see the number of weather measurements per day per station, which shows that a siginficant number of stations have complete data, but the long tail to the left represents the amount of missing values which will need a strategy for handling.
# MAGIC 
# MAGIC <img src="https://github.com/seancampos/w261-fp-images/raw/master/weather_obs_per_day_per_station.jpg" width="600">
# MAGIC 
# MAGIC Additonal details in the EDA [notebook].
# MAGIC 
# MAGIC [notebook]: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/231034620066236/command/231034620066254

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Airlines Data

# COMMAND ----------

#Every row represents one flight. ORIGIN -> DEST
raw_airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/201*.parquet")
print("Number of flights (2015 - 2019):  ", raw_airlines.count())
print("Number of data columns:  ", len(raw_airlines.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop Some Data
# MAGIC We'd like to highlight the count of `NULL` values by column of the airlines data from the get go. The total number of rows in our raw data is `31,746,841`, some columns have nearly `25,000,000+` rows that are `NULL` values. In a scenario where 80% of a column is `NULL`, we have very minimal raw data to use to derive any reasonable imputation method. After learning more about some of these columns from the [documentation], our team came to the conclusion that to minimize noise, dropping these columns was appropriate, and focussing on discovering other signals was the right choice.
# MAGIC 
# MAGIC Dropping irrelevant columns helped trim the horizontal cardinality of our data, but we still had many rows that held `NULL` values. Our approach then was to look at the counts, distributions, and visualize various columns in our data. After learning the general trends of columns in our dataset, we wanted to know if dropping all rows that had even a single `NULL` value would negatively impact the trends we say, for example if there was some bias with rows that had `NULL` values. 
# MAGIC 
# MAGIC We concluded that this was **not** the case, and that by dropping all remaining rows with `NULL` values we did not interfere with existing trends of the data. Therefore we dropped **575,642 rows** which is **1.8132%** of our raw airlines data. This was a tradeoff we were willing to make, losing a very minimal amount of data in exchange for reducing noise and making data processing much easier.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC This and further EDA linked [here] describes the decisions to drop this data.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC [here]: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/231034620064198/command/231034620067191
# MAGIC [documentation]: https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time

# COMMAND ----------

# DBTITLE 1,Airlines Column Name | Null Count 
#COL_NAME | Number of NULLs
raw_null_counts_pd = raw_airlines.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in raw_airlines.columns]).toPandas()
raw_null_counts_pd = raw_null_counts_pd.T.rename(columns={0: "null_count"}) 
raw_null_counts_pd = raw_null_counts_pd.sort_values(by=['null_count'], ascending = False)
raw_null_counts_pd.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enrich Airlines (Time)
# MAGIC In order for our airlines data to join with the weather data, we need to add handle all flight times as **Coordinated Universal Time (UTC)** timestamps. First we need to account for the time zone of every origin and destination airport. From there for each row we can use the respective origin/destionation timezone, as well as the departure and arrival time of the flight, to add columns that represent these times in UTC.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The table `all_airport_stations` follows a similar schema to the stations tables discussed previously. Here we'd like to showcase the use of the **TimezoneFinder** library in a custom **user defined function (UDF)** to add a timezone column to our airport stations data. This eventually gets joined with the remaining flight information as an aid to create the UTC timestamp columns.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```
# MAGIC     def find_tz(lat, lng):
# MAGIC       tf = TimezoneFinder()
# MAGIC       return tf.timezone_at(lat=float(lat), lng=float(lng))
# MAGIC 
# MAGIC     find_tz_udf = f.udf(lambda lat,lng: find_tz(lat, lng), StringType())
# MAGIC     all_airport_stations_tz = all_airport_stations.withColumn('station_tz', find_tz_udf(col('lat'), col('lon')))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ultimately we perform a join to get timezone zone columns into the airlines data. Using several other UDFs we accomplish the addition of UTC columns to our dataset. The work for this can be found [here] 
# MAGIC 
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2127515060477989/command/2127515060480531

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airlines UTC Final
# MAGIC 
# MAGIC After some additional cleaning of outliers, and enrichment, we built an airlines table ready for joins using UTC and ORIGIN/DEST IATA codes as a foreign keys. As a quick comparison, you'll notice that we have lost some more rows. Given the grand scheme of our data, we felt that the amount of data lost is very marginal, and in fact will benefit the model training as we work with cleaner data and reduce noise.
# MAGIC 
# MAGIC * Raw => 31,746,841 rows
# MAGIC * UTC Latest => 31,171,199 rows
# MAGIC 
# MAGIC * **Rows Lost => 575,642 **
# MAGIC * **Rows Percent Loss => 1.8132%**
# MAGIC * **Columns Lost => 43**
# MAGIC * **Columns Percent Loss => 39.449%**
# MAGIC 
# MAGIC 
# MAGIC Data can be accessed at the following path `dbfs:/mnt/mids-w261/team20SSDK/cleaned_data/airlines/airlines_latest_utc/`

# COMMAND ----------

airlines_utc = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/cleaned_data/airlines/airlines_latest_utc/part-00*.parquet")
print("Number of cleaned airlines rows", airlines_utc.count())
print("Number of data columns:  ", len(airlines_utc.columns))
display(airlines_utc)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airlines Charts
# MAGIC 
# MAGIC Below are some graphs that we derived from our cleaned up airlines table. The data that is summarized by these charts reflects all years **2015-2019**.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### CRS Departure Hour - Bar Chart
# MAGIC The hour marks reflect hours local to each timezone. As we can see there is a clear surge in the **morning**, as well as in the **evening**. There is a distinct drop off as we get into the very late hours or very early hours of the day. Local DEP_HOUR can be useful for understanding how passengers and crew are experiencing the airport. 

# COMMAND ----------

displayHTML("<img src ='https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/g_dep_hour2.png?token=AEALQT332CTYLFHLCA7JHNS72RASY' width=1500 height=400>")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### DEP_DEL15 Counts
# MAGIC This chart shows the clear imbalance in our output label column. There is an **~80-20** split as per the amount of on time flights vs. the amount of delayed flights. We will cover how we have accounted for this through class weightage in a later section.
# MAGIC 
# MAGIC * 0 => On Time
# MAGIC * 1 => Delayed

# COMMAND ----------

displayHTML("<img src ='https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/g_dep_del.png?token=AEALQT3M3756XU3MJUGQLKC72RCGY' width=1500 height=400>")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### CRS Departure UTC_HOUR - Bar Chart
# MAGIC The hour marks reflect the hour in **UTC** that a flight was scheduled to depart. This snapshot is a summary for any given hour reflects how many flights across entire country are scheduled to depart. UTC DEP_HOUR can be useful for understanding inter-airport coordination.

# COMMAND ----------

displayHTML('<img src="https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/g_origin_utc2.png?token=AEALQT5BSBP723BP7FTVMNK72RDBQ" width=1500 height=400>')

# COMMAND ----------

# Airlines Table
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/cleaned_data/airlines/airlines_final")
display(airlines)
airlines.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

# COMMAND ----------

# MAGIC %md
# MAGIC As a frequent flyer, we know that flight departures (and arrivals)  often get affected by weather conditions, so it makes sense to collect and process weather data corresponding to the origin and destination airports at the time of departure and arrival respectively and build features based upon this data. 
# MAGIC A weather table  has been pre-downloaded from the National Oceanic and Atmospheric Administration repository  to S3 in the form of  parquet files (thereby enabling pushdown querying and efficient joins). The weather data is for the period Jan 2015 – December 2019.
# MAGIC 
# MAGIC - Note - For a detailed description of the weather EDA, click [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/667068395499598/command/667068395499645). The section below is only a high-level overview of the analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Intial Weather Join
# MAGIC We start with by reading the raw weather information and then joining them with the stations table with a goal to restrict the analysis to US stations only.
# MAGIC As with any real-world dataset, this collected data is messy and has a lot of missing values, which we intend to impute(pls refer to the imputation section)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data section- Mandatory V/S additional 
# MAGIC 
# MAGIC The Integrated Surface Dataset (ISD) is composed of worldwide surface weather observations from over 35,000 stations parameters included are: air quality, atmospheric pressure, atmospheric temperature/dew point, atmospheric winds, clouds, precipitation, ocean waves, tides and more. 
# MAGIC 
# MAGIC We can broadly classify the elements of the dataset into two main categories:
# MAGIC - Mandatory data =>The mandatory data section contains meteorological information on the basic elements such as winds, visibility, and temperature.
# MAGIC These are the most commonly reported parameters and are available most of the time.
# MAGIC - Additional Data Section => These additional data contain information of significance and/or which are received with varying degrees of frequency.
# MAGIC 
# MAGIC For the purpose of this study :
# MAGIC - We will limit the weather observations to the stations in the US only.
# MAGIC - We'll focus on mandatory data section (see the detailed notebook for the additional data section)
# MAGIC - We'll ignore the 'remarks section as ,as these are a set of characeters in plain language that do not provide much insight into decision making.

# COMMAND ----------

# ORIGINAL WEATHER DATA
weather = spark.read.option("header", "true")\
                    .parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/weather_data/*.parquet")


weather = weather.select(year(col("DATE")).alias("YEAR"), month(col("DATE")).alias("MONTH"), dayofmonth(col("DATE")).alias("DAY_OF_MONTH"), concat(rpad(lpad(hour(col("DATE")), 2, '0'), 4, '0'), lit('-'), lpad(hour(col("DATE")), 2, '0'), lit('59')).alias('HOUR_BLOCK'), *weather)

stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/DEMO8/gsod/stations.csv.gz")

#performing the final join on weather,stations and applying filters to restrict it to  US stations only
cleaned_weather = weather.join(stations, [concat(col("usaf"), col("wban")) == weather.STATION, col('country') == "US"]).filter(weather.DATE.between(to_timestamp(stations.begin, 'yyyyMMdd'), to_timestamp(stations.end, 'yyyyMMdd') + expr("INTERVAL 24 hours"))).select(*weather,stations.country)

cleaned_weather.display()

# The code below creates a new dataframe to seperate out 2019 recs
weather.registerTempTable('weather')
stations.registerTempTable('stations')


'''
* Issue was noticed for weather data in 2019
* Within the stations dataset, the max end for 2019 is March 4th
* Initially, in the above query, we were removing weather observations for the latter part of 2019 (therefore were only working with Winter months data)
* We now remove the filter for the active period of a station and retrieve all US weather observations for 2019
'''
cleaned_weather_19 = spark.sql("""
SELECT weather.*, stations.country
FROM weather, stations
WHERE weather.YEAR==2019 and concat(stations.usaf, stations.wban) == weather.STATION AND stations.country == 'US' """)
display(cleaned_weather_19)
cleaned_weather_19.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Really Big Data!!

# COMMAND ----------

# DBTITLE 1,Untitled
cleaned_weather[['YEAR']].display()


# COMMAND ----------

# MAGIC %md
# MAGIC ** Observations **
# MAGIC - We have huge numbe of rows
# MAGIC - Total aggregated data across all the years ~230M

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional data section
# MAGIC We further filter out the records that correspond to the 'additional data'. This is largely due to the fact that the the vast majority of the observations are missing and we belive that it'll violate the principal of parsimony.

# COMMAND ----------

# MAGIC %md
# MAGIC **Proportion of 'Additional-Data' weather metrics that have missing values

# COMMAND ----------

# DBTITLE 1,Count of Missing Values for 'Additional Data' section
displayHTML('<img src="https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/DeveshEDAImages/Weather-Additional_Data_missing_vals.PNG?token=AEALQTZZESGACSNMU6WH74S73LB7M" width=500,height=100>')

# COMMAND ----------

# MAGIC %md
# MAGIC **Observations**
# MAGIC - As we notice that the vast majority of the data in these attributes are missing, we have decided to eliminate these columns.
# MAGIC - This is to ensure that any models and analysis are according to the principal of parsimony.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data format
# MAGIC Each (mandatory) weather observation has a number of attributes associated with it. Below is a brief description of how this information is reprsented in the data.
# MAGIC - Wind
# MAGIC   - Angle
# MAGIC   - Quality of observation
# MAGIC   - Observation type
# MAGIC   - Wind Speed
# MAGIC   - Quality of Wind Speed's observation
# MAGIC 
# MAGIC - Sky Conditions
# MAGIC   - Height
# MAGIC   - Quality of height observation
# MAGIC   - Method used to determine the Ceiling height
# MAGIC   - Code indicating whether ceiling and visibility are OK(CAVOK code)
# MAGIC   - Quality of CAVOK observation
# MAGIC   
# MAGIC - Visibility Observations
# MAGIC   - Distance
# MAGIC   - Quality of observation
# MAGIC   - Variation in visibility
# MAGIC   - Quality code to denote the variation in visibility
# MAGIC   
# MAGIC - Air Temperature
# MAGIC   - Temperature(in degrees celcius)
# MAGIC   - Quality of observed temperature
# MAGIC 
# MAGIC - Dew Temperature
# MAGIC   - Dew point temperature
# MAGIC   - Quality of observed temperature
# MAGIC   
# MAGIC - Atmospheric Pressure
# MAGIC   - Sea Level Pressure
# MAGIC   - Observation quality
# MAGIC 
# MAGIC   
# MAGIC    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis of weather observations.
# MAGIC Note- The weather observations corresponding to a specific weather phenomena are lumped together in their corresponding columns with comma separated values.
# MAGIC The code below splits these observations into individual metrics.

# COMMAND ----------

#splitting the observations- WIND
split_col = f.split(cleaned_weather['WND'], ',')
cleaned_weather = cleaned_weather.withColumn('WND_Angle', split_col.getItem(0).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('WND_Qlty', split_col.getItem(1))
cleaned_weather = cleaned_weather.withColumn('WND_Obs', split_col.getItem(2))
cleaned_weather = cleaned_weather.withColumn('WND_Speed', split_col.getItem(3).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('WND_Speed_Qlty', split_col.getItem(4))

#Splitting 2019 data on WND
split_col = f.split(cleaned_weather['WND'], ',')
cleaned_weather_19 = cleaned_weather_19.withColumn('WND_Angle', split_col.getItem(0).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('WND_Qlty', split_col.getItem(1))
cleaned_weather_19 = cleaned_weather_19.withColumn('WND_Obs', split_col.getItem(2))
cleaned_weather_19 = cleaned_weather_19.withColumn('WND_Speed', split_col.getItem(3).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('WND_Speed_Qlty', split_col.getItem(4))

#Sky observations
split_col = f.split(cleaned_weather['CIG'], ',')
cleaned_weather = cleaned_weather.withColumn('CIG_Height', split_col.getItem(0).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('CIG_Qlty', split_col.getItem(1))
cleaned_weather = cleaned_weather.withColumn('CIG_Ceiling', split_col.getItem(2))
cleaned_weather = cleaned_weather.withColumn('CIG_CAVOK', split_col.getItem(3))


# 2019 SKY observations
split_col = f.split(cleaned_weather['CIG'], ',')
cleaned_weather_19 = cleaned_weather_19.withColumn('CIG_Height', split_col.getItem(0).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('CIG_Qlty', split_col.getItem(1))
cleaned_weather_19 = cleaned_weather_19.withColumn('CIG_Ceiling', split_col.getItem(2))
cleaned_weather_19 = cleaned_weather_19.withColumn('CIG_CAVOK', split_col.getItem(3))


#Visibility conditions

split_col = f.split(cleaned_weather['VIS'], ',')
cleaned_weather = cleaned_weather.withColumn('VIS_Dis', split_col.getItem(0).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('VIS_Qlty', split_col.getItem(1))
cleaned_weather = cleaned_weather.withColumn('VIS_Var', split_col.getItem(2))
cleaned_weather = cleaned_weather.withColumn('VIS_Var_Qlty', split_col.getItem(3))

# 2019 visibility data
split_col = f.split(cleaned_weather['VIS'], ',')
cleaned_weather_19 = cleaned_weather_19.withColumn('VIS_Dis', split_col.getItem(0).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('VIS_Qlty', split_col.getItem(1))
cleaned_weather_19 = cleaned_weather_19.withColumn('VIS_Var', split_col.getItem(2))
cleaned_weather_19 = cleaned_weather_19.withColumn('VIS_Var_Qlty', split_col.getItem(3))


# TMP
split_col = f.split(cleaned_weather['TMP'], ',')
cleaned_weather = cleaned_weather.withColumn('TMP_Degree', split_col.getItem(0).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('TMP_Qlty', split_col.getItem(1))

# 2019 TMP
split_col = f.split(cleaned_weather['TMP'], ',')
cleaned_weather_19 = cleaned_weather_19.withColumn('TMP_Degree', split_col.getItem(0).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('TMP_Qlty', split_col.getItem(1))

#Dew
split_col = f.split(cleaned_weather['DEW'], ',')
cleaned_weather = cleaned_weather.withColumn('DEW_Degree', split_col.getItem(0).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('DEW_Qlty', split_col.getItem(1))

split_col = f.split(cleaned_weather['DEW'], ',')
cleaned_weather_19 = cleaned_weather_19.withColumn('DEW_Degree', split_col.getItem(0).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('DEW_Qlty', split_col.getItem(1))


#Pressure
split_col = f.split(cleaned_weather['SLP'], ',')
cleaned_weather = cleaned_weather.withColumn('SLP_Pressure', split_col.getItem(0).cast(IntegerType()))
cleaned_weather = cleaned_weather.withColumn('SLP_Qlty', split_col.getItem(1))

#2019 pressure

split_col = f.split(cleaned_weather['SLP'], ',')
cleaned_weather_19 = cleaned_weather_19.withColumn('SLP_Pressure', split_col.getItem(0).cast(IntegerType()))
cleaned_weather_19 = cleaned_weather_19.withColumn('SLP_Qlty', split_col.getItem(1))

#Dropping the redundant cols that had the comma separated values
cleaned_weather_final = cleaned_weather.drop("WND","CIG","VIS","TMP","DEW","SLP")
cleaned_weather_19_final = cleaned_weather_19.drop("WND","CIG","VIS","TMP","DEW","SLP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing values Analysis
# MAGIC Like any hardware device the weather station/sensors are subject to failures.This results is observations that are missing or are of poor quality,This phenomena(device-failure)/missing observation is very common in real workd and any modeling we do should be capable of handling missing values.
# MAGIC In the weather data the missing observations are denoted by 'out-of range' values like (999 etc.) . The below chart represents the missing values in our training data and this inturn gives us an indication of the extent of imputation that we have to do in the data frame.

# COMMAND ----------

# DBTITLE 1,Missing values by year
displayHTML('<img src="https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/DeveshEDAImages/Weather_imputation_missing_vals.PNG?token=AEALQTZ3N4UTAN4XCRZ3RZC73LCBE" width=500,height=100>')

# COMMAND ----------

# MAGIC %md
# MAGIC **Observations**
# MAGIC - We need to impute almost 55 million rows across all the years for the 'mandatory data columns'
# MAGIC 
# MAGIC Please note - The actual imputations might be less as a result of the final joins with the airline data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Engineering
# MAGIC In this section we will summarize and highlight the features that we derived and built from the provided data. For each feature we have a separate notebook that goes into greater detail linked respectively. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Pagerank
# MAGIC 
# MAGIC ##### Theory:
# MAGIC We are interested in engineering features that will help our model predict how flight delays propagate through the airport network causing additional downstream delays.  To this end, the Pagerank algorithm, which is normally used to quantify the significance of web pages, also has a number of qualities which provide valuable analogs to our problem at hand.
# MAGIC 
# MAGIC We would like to quantify both how likely an airport is to be subject to incoming arrival delays, and how much of an effect any outgoing delays will have on the rest of the transportation network.  Pagerank is typically calculated as a markov chain by imaging the act of navigating the internet to be a random walk.  Each page is initialized with equal probability of being reached, then that probability is iteratively updated by distributing it among the outgoing links from that page and aggregating it from the incoming links to that page until steady state is reached. 
# MAGIC 
# MAGIC A similar analogy applies to the airport network, substituting incoming links for arriving flights and outgoing links as departing flights.  In this way we can quantify the probability of an airport node being involved in a propagated flight delay.  While this is closely related to other network graph metrics, such as eigenvector centrality, Pagerank’s treatment of directed weighted graphs makes it a more representative choice for our modeling goals.
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Use the NetworkX Graph library to analyze the underlying flow of traffic at all US airports via Pagerank.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC By calculating the **Pagerank** (popularity) of an airport, we are able to capture the network flow that exists amongst airports in the US. This score would reflect the random chance of traveller ending up at a given airport. We hope to see a meaningful correlation between Pagerank and `DEP_DEL15` whether that is positive or negative. As we get to modelling stages, we will experiment with `ORIGIN_PAGERANK` and `DEST_PAGERANK` both as features.
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2127515060476900/command/4061768275137079

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Group all `ORIGIN -> DEST` flight paths and get the counts**
# MAGIC 
# MAGIC We will use these counts as the weights on this directed graph. Where the weight of an edge from node A to Node B is the number of times a flight has made this trip in the time range 2015-2018.
# MAGIC 
# MAGIC ```
# MAGIC grouped_by_trip = airlines_pd.groupby(['ORIGIN','DEST']).size().reset_index()
# MAGIC grouped_by_trip = grouped_by_trip.rename(columns={0: "count"})
# MAGIC ```
# MAGIC 
# MAGIC |      | ORIGIN   | DEST   |   count |
# MAGIC |-----:|:---------|:-------|--------:|
# MAGIC |    0 | ABE      | ATL    |    3790 |
# MAGIC |    1 | ABE      | CLT    |     911 |
# MAGIC |    2 | ABE      | DTW    |    3201 |
# MAGIC |    3 | ABE      | EWR    |       1 |
# MAGIC |    4 | ABE      | FLL    |      72 |
# MAGIC |    5 | ABE      | MYR    |      84 |
# MAGIC |    6 | ABE      | ORD    |    2104 |
# MAGIC |    7 | ABE      | PGD    |     144 |
# MAGIC |    8 | ABE      | PHL    |      40 |
# MAGIC |    9 | ABE      | PIE    |     157 |
# MAGIC |   10 | ABE      | SFB    |     440 |

# COMMAND ----------

displayHTML('<img src="https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/g_pageranks.png?token=AEALQT3QYJS22LJB5QV6U4S727B32" width=900 height=350>')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Delays so Far
# MAGIC 
# MAGIC As shown through research papers, as well as personal flying experience, if an airport is experiencing delays on a given morning, this delay will propagate throughout the day to impact later flights. We would like to capture this information explicitly associated by flight **TAIL_NUM**
# MAGIC 
# MAGIC ##### Goal:
# MAGIC For a given flight, represented by TAIL_NUM, figure out the number of times that same TAIL_NUM was delayed previously on that day up to 2 hours before the given flight.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC If I am to fly out of LAX at 7PM to SFO. For my flights given TAIL_NUM, which is a unique identifier for that flight route (origin -> destination) if that TAIL_NUM is experiencing flight delays anytime prior to the time of my flight on that same day, there is a chance that this will cause my flight to be delayed as well.
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC 
# MAGIC Below is a groupby on the number of `delays_so_far` that occur across all years 2015-2019. As we can see the largest group is no `delays_so_far` and this lines up with the fact that 80% of our flights are on time. Even if there were `delays_so_far` there exists chance that the delay can be recoverred during the flight.
# MAGIC 
# MAGIC | delays_so_far | count    |
# MAGIC |--------------:|----------|
# MAGIC |           0.0 | 24077110 |
# MAGIC |           1.0 |  5065569 |
# MAGIC |           2.0 |  1423046 |
# MAGIC |           3.0 |   431950 |
# MAGIC |           4.0 |   125448 |
# MAGIC |           5.0 |    34714 |
# MAGIC |           6.0 |     8652 |
# MAGIC |           7.0 |     1659 |
# MAGIC |           8.0 |      288 |
# MAGIC |           9.0 |       37 |
# MAGIC |          10.0 |       16 |
# MAGIC |          11.0 |        1 |
# MAGIC |          12.0 |        5 |
# MAGIC |          14.0 |        1 |
# MAGIC 
# MAGIC #### Label Percentages
# MAGIC * 0 --> On Time
# MAGIC * 1 --> Delayed
# MAGIC 
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/delays_percent.png?token=AEALQT7K37ZA4LAYGOTOBIC73P3IG>
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/3450569410566534/command/4061768275135506

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Flight Count
# MAGIC 
# MAGIC ##### Goal:
# MAGIC On a per day basis compute the total number of flights departing, and arriving at a given airport.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC The number of flights to/from an airport tells us how busy that airport is on a given day. This feature would be very correlated to the size/popularity of an airport, we hope to see if there is a strong correlation between this and predicting `DEP_DEL15`
# MAGIC 
# MAGIC Note: Our data only deals with flights that were neither diverted nor cancelled.
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2127515060477729/command/4061768275137695

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Network Congestion
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Use the NetworkX Graph library to find the ten most central nodes in the airport transportation network.  Count how many delays have occurred across all of these airports on the same day two hours prior to any flight.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC With the ‘Delays so Far’ feature we looked directly at the daily history of a particular aircraft to see if it has been impacted by a delay today, which contains a great deal of relevant information, however with the large number of craft in service at any given time, this metric is capturing a great deal of variance could lead to 'memorizing' different aircraft and thus overfitting.  One way to capture similar information, but with less variability and more bias is to focus only on the delays associated with the core of the transportation network.
# MAGIC 
# MAGIC To achieve this we take a similar approach, but count the number of delays prior to two hours before departure for the top ten most central airports to the network graph. In graph theory, centrality is a measure of importance, and the type of importance that we are most interested in is betweenness centrality.  It is the sum of the fraction of shortest paths between any two vertices that pass through a given vertex.  This is an analog to the hub and spoke model that the airport network uses to route flights between smaller regional airports through larger hub airports, and the airlines are incentivized to use the shortest path to save fuel costs.  The nodes with the highest scores are likely to precede other routes, thus delays at these airports could affect seemingly unrelated flights.
# MAGIC 
# MAGIC This gives us a window into the higher-level state of the flight network and provides the same value to all flights at the same time regardless of location, which in the end should give us a metric that is less likely to overfit our training data.
# MAGIC 
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/3450569410586574/command/4061768275142078
# MAGIC 
# MAGIC <img src="https://github.com/seancampos/w261-fp-images/raw/master/network_congestion_delay_performance.png">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### West to East
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Use the longitudes of the origin and destination airport to determine if the flight is going from West to East.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC We have constructed a number of features which focus on detecting propagation delay in the airline travel network, however delays can also be recovered from and we want to attempt to measure that likelihood as well.  The jet stream blows eastward and often provides a tail wild which allows aircraft to increase their ground speed without incurring higher fuel costs, which are typically a limiting factor imposed on pilots by airline policy.  The West to East feature identifies flight paths which are traveling in the same direction as the jet stream by comparing origin and destination longitudes and have a higher probability of recovering from a prior delay.
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275142113/command/2834511320013375
# MAGIC 
# MAGIC <img src="https://github.com/seancampos/w261-fp-images/raw/master/west_to_east_performance.png">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### CRS Elapsed Time Average Difference
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Compare the scheduled flight duration to the average duration for that route.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC The number of air travel passengers has been reaching all time highs for the past few years, and in an effort to maximize potiential revenue, airlines have been known to try to fit more flights into their schedule.  This reduces the amount of 'buffer time' between flights that is availble to absorb small delays, or to allow the schedule to recover from a previous delay.  To catpture this information we measured the average scheduled duration for each origin-destination pair and then created a feature that is the difference between the current scheduled flight and the average. If the airlines are giving the current flight less time than is typical, we will see a negative number, and if it's longer than average we'll get a positive number.
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/3450569410578953/command/2834511320013422
# MAGIC 
# MAGIC <img src="https://github.com/seancampos/w261-fp-images/raw/master/crs_elapsed_time_avg_diff_performance.png">

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Departure/ Arrival Hour Bins
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Improve parsimony and prevent overfitting by dividing the day into discrete blocks of time instead using the specific departure time.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC The time of day that a flight departs has a clear relationship with its likelihood of being delayed.   The longer a day goes on, the more possibilites of things going wrong arise.  However, this behavior does not change much with blocks of several hours, and treating each hour as categorical varible creates 24 additional colunms.  By binning together hours that have the same behavior improves parsimony and helps to guard against overfitting.  The chart below shows the individual hour departure delay performance.  There are clear sections with similar behavior such as the 0-5 hours.
# MAGIC 
# MAGIC Sample Query For Hour Intervals:
# MAGIC ```
# MAGIC SELECT data_holiday_week_15_to_18.*, CASE WHEN DEP_HOUR < 6 THEN 0
# MAGIC        WHEN DEP_HOUR >= 6 AND DEP_HOUR <= 9 THEN 1
# MAGIC        WHEN DEP_HOUR >= 10 AND DEP_HOUR <= 14 THEN 2
# MAGIC        WHEN DEP_HOUR >= 15 AND DEP_HOUR <= 20 THEN 3
# MAGIC        WHEN DEP_HOUR > 20 THEN 4 
# MAGIC        END AS DEP_HOUR_BIN
# MAGIC FROM data_holiday_week_15_to_18
# MAGIC ```
# MAGIC 
# MAGIC Detailed notebook [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275129334/command/4061768275130894)
# MAGIC 
# MAGIC <img src="https://github.com/seancampos/w261-fp-images/raw/master/dep_hour_bins_performance.png.png">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Holiday Weeks
# MAGIC 
# MAGIC ##### Goal:
# MAGIC From observations we know that during holiday weeks, there is a surge in the passenger traffic and airlines add extra capacity to meet this additional demand.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC The number of air travel passengers has been reaching all time highs for the past few years, and in an effort to maximize potiential revenue, airlines have been known to try to fit more flights into their schedule. This is especially true during holidays(weekends around holidays).
# MAGIC Our estimate is that this added capacity results in additional overhead due to increased network congestion.
# MAGIC 
# MAGIC Detailed notebook [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2834511320012837/command/2834511320012840)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Minutes After Midnight
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Create a numerical representation of the flight departure / arrival times. This measurement is based off the UTC-converted times.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC By providing more granularity of how time is represented in our model and by treating these values numerically, we can scale these values accordingly and hope to see a greater correlation between this feature and predicting `DEP_DEL15`. This is more parsimonious than treating each hour as a categorical variable.  Since we’re using decisions trees, the nonlinear behaviour association between the feature value an the delay likelihood can be captured by the model.
# MAGIC 
# MAGIC 
# MAGIC <img src='https://github.com/seancampos/w261-fp-images/raw/master/mins_after_midnight_performance.png' height="400">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Is Morning Flight, Is Evening Flight
# MAGIC 
# MAGIC ##### Goal:
# MAGIC Create binary features to represent whether a flight has departed in the morning or the evening.
# MAGIC 
# MAGIC ##### Hypothesis:
# MAGIC These features are created from the departure hour bins. Specifically, we noticed that **bin 1** (6:00am - 9:00am) and **bin 3** (3:00 - 8:00pm) showcase increased spikes in the number of delayed flights during those time intervals. These time intervals may even typically be considered commuter hours and by once more reducing granularity for departure times, we hope to reduce overfitting and allow for the model to learn from the behavior of flight delays during these periods of time.
# MAGIC 
# MAGIC 
# MAGIC Detailed notebook [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275129334/command/2834511320013420)
# MAGIC 
# MAGIC 
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/depHourbins2.png?token=AEALQTYXDLPVWIZXBZ7LYNS73U7XE>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Station Nearest Neighbors
# MAGIC 
# MAGIC One of the significant challenges with the dataset is that the weather metrics reported from the stations located at the airports is quite frequently missing or unusable.  One remedy we sought for this was to look to nearby stations, since weather is spatially localized, and at the speed planes travel at there is a considerable radius of weather conditions relevant to landing and takeoff operations. The challenge is that there are many thousands of weather stations, and finding all of the stations that are within the range of interest for each airport would involve a very large number of possible permutations of stations. 
# MAGIC 
# MAGIC Fortunately, this can be formulated as an embarrassingly parallel problem suitable for the MapReduce framework.  The distance calculation between each station is performed by the haversine formula, which only depends on the coordinates of the two stations which the distance is being measured between, so it can be evenly spread across as many mappers as available.  The resulting distance is emitted with a compound key consisting of the airport's station identifier, followed by the target weather station.  The shuffle stage then naturally brings each airport's neighbors onto the same partition and orders them by the closest distance, which is exactly the order in which we want to try to fill in missing or invalid weather measurements.
# MAGIC 
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/231034620066236/command/231034620066549)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Code Analysis
# MAGIC 
# MAGIC 
# MAGIC > Compute all possible permuations of station pairs and create an RDD of the data
# MAGIC 
# MAGIC ```
# MAGIC coords = []
# MAGIC 
# MAGIC for long, lat, station_id in zip(locations['lat'], locations['lon'], locations['station_id']):
# MAGIC     coords.append((float(lat),float(long), station_id))
# MAGIC 
# MAGIC coords_list = []
# MAGIC 
# MAGIC for combo in permutations(coords, 2):
# MAGIC     coords_list.append(combo)
# MAGIC 
# MAGIC coords_rdd = sc.parallelize(coords_list)
# MAGIC ```
# MAGIC 
# MAGIC > Create a mapreduce job to map through each record in the `coords_rdd` and create a tuple to map each origin station to all other stations with the distances between each. Then, reduce all tuples to combine all dictionaries of neighboring stations for each origin station.
# MAGIC 
# MAGIC ```
# MAGIC def calcHaversine(x):
# MAGIC     origin, dest = x[0], x[1]
# MAGIC     dest_dict = {}
# MAGIC     dist = haversine(origin[:2], dest[:2], unit=Unit.MILES)
# MAGIC     dest_dict[dest] = dist
# MAGIC     return (origin, dest_dict)
# MAGIC   
# MAGIC def updateDict(x, y):
# MAGIC     x.update(y)
# MAGIC     return x
# MAGIC 
# MAGIC result = coords_rdd.map(lambda x: calcHaversine(x)) \
# MAGIC                    .reduceByKey(lambda x, y: updateDict(x, y)).cache()
# MAGIC ```
# MAGIC 
# MAGIC > Now that we have computed the distances from each origin station to all other stations, filter to only get all neighboring stations within a 50 mile radius of each origin station. The final result is a mapping of each origin station to itself as well as each of its neighboring stations.
# MAGIC 
# MAGIC ```
# MAGIC def getStationNeighbors(x):
# MAGIC     origin, dest = x[0], x[1]
# MAGIC     yield (origin[2], origin[2])
# MAGIC     sorted_list = [str(i[0][2]) for i in filter(lambda x: x[1] <= 50.0, sorted(dest.items(), key=lambda x: x[1]))]
# MAGIC     for i in sorted_list:
# MAGIC         yield(origin[2], i)
# MAGIC 
# MAGIC clusters = result.flatMap(lambda x: getStationNeighbors(x)).cache()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Dataset

# COMMAND ----------

station_neighbors = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/cleaned_data/station/station_neighbors")
display(station_neighbors)

station_neighbors.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### SNN Map
# MAGIC 
# MAGIC Following is a map of the US in which all of the airport origin stations have been plotted. Hovering over them will describe how many neighboring stations are located within a 50 mile radius.

# COMMAND ----------

station_neighbors.registerTempTable('station_neighbors')
station_neighbors_count = spark.sql("""
SELECT station_id, count(neighbor_id) as NUM_NEIGHBORS FROM station_neighbors GROUP BY station_id
""")

display(station_neighbors_count)

# COMMAND ----------

airport_meta = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/cleaned_data/station/airport_meta")

display(airport_meta)
airport_meta.count()

# COMMAND ----------

airport_meta.registerTempTable('airport_meta')
station_neighbors_count.registerTempTable('station_neighbors_count')

airport_meta_nn = spark.sql("""
SELECT a.*, snc.NUM_NEIGHBORS
FROM airport_meta as a LEFT JOIN station_neighbors_count as snc
  ON snc.station_id == a.STATION
""")

airport_meta_pandas = airport_meta_nn.toPandas()
airport_meta_pandas

# COMMAND ----------

# DBTITLE 1,US Airport Weather Stations
fig = go.Figure(data=go.Scattergeo(
        lon = airport_meta_pandas['lon'],
        lat = airport_meta_pandas['lat'],
        text = "Name: " + airport_meta_pandas['name'].astype(str) + "<br>Neighbors: " + airport_meta_pandas['NUM_NEIGHBORS'].astype(str),
        mode = 'markers',
))

fig.update_layout(
        title = 'US Airport Weather Stations',
        width = 1500,
        height = 1000,
        geo = dict(bgcolor= 'rgba(0,0,0,0)'),
        paper_bgcolor='#4E5D6C',
        plot_bgcolor='#4E5D6C',
        geo_scope ='usa',
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Creating The Final Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Goal / Overview
# MAGIC 
# MAGIC Combine Airlines data with Weather data to map each flight record with weather observations that are at least 2 hours prior to departure time. 
# MAGIC 
# MAGIC In order to achieve our goal, the following procedure will detail our work in creating a `weather_imputed` table of imputed weather measurements and a `weather_metrics` table that consists of average or min weather measurments data as well as a 3-phase imputation approach to handle missing or null data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Column Oriented Data Structure
# MAGIC 
# MAGIC Another course concept that we have religiously followed is to process the data in a column oriented format.
# MAGIC The most common row format is to store the data is in a .txt form and I/O operations on data stored in .txt format from disk results in inefficiencies.
# MAGIC 
# MAGIC The results of our joins are stored in **parquet** format which have proven to be much more efficient compared to row oriented data stores . 
# MAGIC For quick analysis like counts and for reading the data for modelling, the parquet format and the meta data it stores has been the right choice!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Joins Procedure
# MAGIC 
# MAGIC **NOTE**: Following code snippets are the first queries that are run in each respective section. All subsequent queries can be found within the linked notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Create Imputed Weather Table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Imputation via station neighbors for weather measurement columns in the Weather dataset (Phase-1 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT w1.YEAR, w1.MONTH, w1.DAY_OF_MONTH, w1.STATION, w1.DATE, w1.SOURCE, w1.LATITUDE, w1.LONGITUDE, w1.ELEVATION, 
# MAGIC        w1.NAME, w1.REPORT_TYPE, w1.CALL_SIGN, w1.QUALITY_CONTROL, w1.country, avg(w2.WND_Speed) as WND_SPEED
# MAGIC FROM weather as w1, station_neighbors as sn, weather as w2 
# MAGIC WHERE w1.STATION == sn.station_id AND w2.YEAR == w1.YEAR AND w2.MONTH == w1.MONTH AND w2.DAY_OF_MONTH == w1.DAY_OF_MONTH AND EXTRACT(HOUR FROM w2.DATE) == EXTRACT(HOUR FROM w1.DATE) AND 
# MAGIC       w2.STATION == sn.neighbor_id AND w2.WND_Speed_Qlty NOT IN ('2', '3', '6', '7') AND (w1.WND_Speed = 9999 OR w1.WND_Speed is null) AND (w2.WND_Speed != 9999 AND w2.WND_Speed is not null)
# MAGIC GROUP BY w1.YEAR, w1.MONTH, w1.DAY_OF_MONTH, w1.STATION, w1.DATE, w1.SOURCE, w1.LATITUDE, w1.LONGITUDE, w1.ELEVATION, w1.NAME, w1.REPORT_TYPE, w1.CALL_SIGN, w1.QUALITY_CONTROL, w1.country
# MAGIC                         
# MAGIC UNION
# MAGIC 
# MAGIC SELECT w1.YEAR, w1.MONTH, w1.DAY_OF_MONTH, w1.STATION, w1.DATE, w1.SOURCE, w1.LATITUDE, w1.LONGITUDE, w1.ELEVATION, 
# MAGIC        w1.NAME, w1.REPORT_TYPE, w1.CALL_SIGN, w1.QUALITY_CONTROL, w1.country, w1.WND_Speed
# MAGIC FROM weather as w1, (SELECT distinct neighbor_id FROM station_neighbors) as sn
# MAGIC WHERE w1.WND_Speed != 9999 AND w1.WND_Speed is not null AND w1.WND_Speed_Qlty NOT IN ('2', '3', '6', '7') AND w1.STATION == sn.neighbor_id
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Ultimately, it is important to note in our final join between airlines and weather, we are only interested in mapping airport weather stations with its respective weather observations. As a result, we use the above query to ensure that only airport stations and their neighboring stations are accounted for in the weather dataset.
# MAGIC 
# MAGIC The top half of the query represents an inner join between the Weather dataset (for origin stations), station neighbors dataset (mapping of origin stations with their neighbors), and the Weather dataset again (for neighboring stations). Specifically, we select all of the necessary Weather columns and aggregate all of the neighboring measurments for each weather measurement column by taking the average of the neighbors' measurements and imputing the missing data (ie. 9999) and nulls with the computed average measurements. We add a filter to ensure that we select only the best quality data from the neighboring measurements, that is not of qualities 2, 3, 6, and 7. Additionally, we ensure that we include all measurments from neighboring stations that match within the same hour.
# MAGIC 
# MAGIC Next, we perform a union with weather records in the Weather table that do not require imputation and where stations are either airport stations or its neighbors.
# MAGIC 
# MAGIC **Why did we split the imputation of each weather measurement column into separate queries?**
# MAGIC 
# MAGIC We decided to perform sequential imputation of each of the weather measurement columns because it is not guaranteed that if a given measurement column is missing/null or of bad quality, then the rest of the measurements are also of the same type. So, we had to process each separately to ensure that each was accounted for correctly.
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/1551285172210152/command/1551285172210164)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Join the above imputed tables of weather measurements together to create the imputed weather table
# MAGIC 
# MAGIC ```
# MAGIC SELECT coalesce(w1.YEAR, w2.YEAR) as YEAR, coalesce(w1.MONTH, w2.MONTH) as MONTH, coalesce(w1.DAY_OF_MONTH, w2.DAY_OF_MONTH) as DAY_OF_MONTH, 
# MAGIC        coalesce(w1.STATION, w2.STATION) as STATION, coalesce(w1.DATE, w2.DATE) as DATE, coalesce(w1.SOURCE, w2.SOURCE) as SOURCE, 
# MAGIC        coalesce(w1.LATITUDE, w2.LATITUDE) as LATITUDE, coalesce(w1.LONGITUDE, w2.LONGITUDE) as LONGITUDE, 
# MAGIC        coalesce(w1.ELEVATION, w2.ELEVATION) as ELEVATION, coalesce(w1.NAME, w2.NAME) as NAME, 
# MAGIC        coalesce(w1.REPORT_TYPE, w2.REPORT_TYPE) as REPORT_TYPE, coalesce(w1.CALL_SIGN, w2.CALL_SIGN) as CALL_SIGN, 
# MAGIC        coalesce(w1.QUALITY_CONTROL, w2.QUALITY_CONTROL) as QUALITY_CONTROL, coalesce(w1.country, w2.country) as COUNTRY, w1.WND_SPEED, w2.CIG_HEIGHT
# MAGIC FROM weather_wnd_imp as w1 FULL JOIN weather_cig_imp as w2
# MAGIC ON w1.YEAR == w2.YEAR AND w1.MONTH == w2.MONTH AND w1.DAY_OF_MONTH == w2.DAY_OF_MONTH AND w1.STATION == w2.STATION AND 
# MAGIC    w1.DATE == w2.DATE AND w1.SOURCE == w2.SOURCE AND w1.LATITUDE == w2.LATITUDE AND w1.LONGITUDE == w2.LONGITUDE AND 
# MAGIC    w1.ELEVATION == w2.ELEVATION AND w1.NAME == w2.NAME AND w1.REPORT_TYPE == w2.REPORT_TYPE AND 
# MAGIC    w1.CALL_SIGN == w2.CALL_SIGN AND w1.QUALITY_CONTROL == w2.QUALITY_CONTROL AND w1.country == w2.country
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Now that we have created the separate imputed tables for each of our weather measurement columns, we will then join each of those tables together in a sequential manner to create the imputed Weather table. In order to do so, we perform a full join between each imputed table so that we don't exclude records that exist in only one of these tables. We use the `coalesce` function in SQL to ensure that if a given column is null in the first table, use the associated value specificed in the second table.
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/1551285172210152/command/1551285172210184)

# COMMAND ----------

weather_imputed = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/strategy/weather_imputed")
display(weather_imputed)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Create weather metrics table that consists of average or min measurements for each station by date

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Imputation via station neighbors for weather measurement columns in the Weather dataset
# MAGIC 
# MAGIC ```
# MAGIC SELECT w1.YEAR, w1.MONTH, w1.DAY_OF_MONTH, w1.STATION, w1.DATE, w1.SOURCE, w1.LATITUDE, w1.LONGITUDE, w1.ELEVATION, 
# MAGIC        w1.NAME, w1.REPORT_TYPE, w1.CALL_SIGN, w1.QUALITY_CONTROL, w1.country, avg(w2.WND_Speed) as AVG_WND_SPEED
# MAGIC FROM weather_wnd_imp as w1, station_neighbors as sn, weather_wnd_imp as w2 
# MAGIC WHERE w1.STATION == sn.station_id AND w2.YEAR == w1.YEAR AND w2.MONTH == w1.MONTH AND w2.DAY_OF_MONTH == w1.DAY_OF_MONTH AND EXTRACT(HOUR FROM w2.DATE) == EXTRACT(HOUR FROM w1.DATE) AND 
# MAGIC       w2.STATION == sn.neighbor_id
# MAGIC GROUP BY w1.YEAR, w1.MONTH, w1.DAY_OF_MONTH, w1.STATION, w1.DATE, w1.SOURCE, w1.LATITUDE, w1.LONGITUDE, w1.ELEVATION, w1.NAME, w1.REPORT_TYPE, w1.CALL_SIGN, w1.QUALITY_CONTROL, w1.country
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Using the imputed tables for each weather measurement column, we will then take the average measurement for each station plus its neighbors. This computed average measurement will then replace the imputed values with the calculated averages of imputed measurements. We decided to do this because instead of simply using the measurement from one specific station (which may not always be reliable), taking the average of the current station's measurements as well as its neighbors' allows us to better account for the surrouding areas and increase reliability, while reducing noise. The above query depicts an inner join between the imputed weather measurement dataset with the station neighbors dataset as well as with itself again. Specifically, we include neighboring observations within the same hour for computing our average or min metrics.
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/1551285172210152/command/1551285172210198)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Join the above average/min tables of weather measurements together to create the imputed weather metrics table
# MAGIC 
# MAGIC ```
# MAGIC SELECT coalesce(w1.YEAR, w2.YEAR) as YEAR, coalesce(w1.MONTH, w2.MONTH) as MONTH, coalesce(w1.DAY_OF_MONTH, w2.DAY_OF_MONTH) as DAY_OF_MONTH, 
# MAGIC        coalesce(w1.STATION, w2.STATION) as STATION, coalesce(w1.DATE, w2.DATE) as DATE, coalesce(w1.SOURCE, w2.SOURCE) as SOURCE, 
# MAGIC        coalesce(w1.LATITUDE, w2.LATITUDE) as LATITUDE, coalesce(w1.LONGITUDE, w2.LONGITUDE) as LONGITUDE, 
# MAGIC        coalesce(w1.ELEVATION, w2.ELEVATION) as ELEVATION, coalesce(w1.NAME, w2.NAME) as NAME, 
# MAGIC        coalesce(w1.REPORT_TYPE, w2.REPORT_TYPE) as REPORT_TYPE, coalesce(w1.CALL_SIGN, w2.CALL_SIGN) as CALL_SIGN, 
# MAGIC        coalesce(w1.QUALITY_CONTROL, w2.QUALITY_CONTROL) as QUALITY_CONTROL, coalesce(w1.country, w2.country) as COUNTRY, w1.AVG_WND_SPEED, w2.AVG_CIG_HEIGHT, w2.MIN_CIG_HEIGHT
# MAGIC FROM weather_wnd_avg as w1 FULL JOIN weather_cig_avg_min as w2
# MAGIC ON w1.YEAR == w2.YEAR AND w1.MONTH == w2.MONTH AND w1.DAY_OF_MONTH == w2.DAY_OF_MONTH AND w1.STATION == w2.STATION AND 
# MAGIC    w1.DATE == w2.DATE AND w1.SOURCE == w2.SOURCE AND w1.LATITUDE == w2.LATITUDE AND w1.LONGITUDE == w2.LONGITUDE AND 
# MAGIC    w1.ELEVATION == w2.ELEVATION AND w1.NAME == w2.NAME AND w1.REPORT_TYPE == w2.REPORT_TYPE AND 
# MAGIC    w1.CALL_SIGN == w2.CALL_SIGN AND w1.QUALITY_CONTROL == w2.QUALITY_CONTROL AND w1.country == w2.country
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Now that we have created the separate average/min tables for each of our weather measurement columns, we will then join each of those tables together in a sequential manner to create the imputed avergae/min Weather metrics table. In order to do so, we perform a full join between each of the above measurement tables so that we don't exclude records that exist in only one of these tables. We use the `coalesce` function in SQL to ensure that if a given column is null in the first table, use the associated value specificed in the second table.
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/1551285172210152/command/1551285172210212)

# COMMAND ----------

weather_metrics_imputed = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/strategy/weather_metrics_imputed")
display(weather_metrics_imputed)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Perform final set of joins to combine airlines and weather data
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/1551285172210152/command/1551285172210234)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Add station and adjusted timestamp columns to the airlines dataset
# MAGIC 
# MAGIC ```
# MAGIC airlines = airlines.join(airport_meta.alias('AM'), airlines.ORIGIN == col('AM.IATA')).select(*airlines, col('AM.STATION').alias('ORIGIN_STATION'), col('AM.NAME').alias('ORIGIN_STATION_NAME'), col('AM.pagerank').alias('PAGERANK'))
# MAGIC 
# MAGIC airlines = airlines.join(airport_meta.alias('AM'), airlines.DEST == col('AM.IATA')).select(*airlines, col('AM.STATION').alias('DEST_STATION'), col('AM.NAME').alias('DEST_STATION_NAME'))
# MAGIC 
# MAGIC SELECT a.*, a.ORIGIN_UTC - INTERVAL 50 HOURS as ORIGIN_UTC_ADJ_MIN, a.ORIGIN_UTC - INTERVAL 2 HOURS as ORIGIN_UTC_ADJ_MAX
# MAGIC FROM airlines as a
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Join airlines with the airport_meta dataset matching the ORIGIN / DEST airport column from airlines with the IATA code column from airport_meta to add in the ORIGIN_STATION and DEST_STATION ID columns and other columns including station names and pagerank.
# MAGIC 
# MAGIC Within the airlines dataset, create two additional columns `ORIGIN_UTC_ADJ_MIN` which represents the timestamp 50 hours prior to departure UTC time as well as `ORIGIN_UTC_ADJ_MAX` which represents the timestamp 2 hours prior to departure UTC time. We will use these two columns in subsequent queries to matching flight records with weather observations.
# MAGIC 
# MAGIC > Create the `weather_origin_helper` and `weather_dest_helper` tables
# MAGIC 
# MAGIC ```
# MAGIC SELECT a.ORIGIN_STATION, a.ORIGIN_UTC, max(wo.DATE) as ORIGIN_MAX_DATE
# MAGIC FROM airlines as a, weather_metrics_imputed as wo
# MAGIC WHERE wo.STATION == a.ORIGIN_STATION AND wo.DATE BETWEEN a.ORIGIN_UTC_ADJ_MIN AND a.ORIGIN_UTC_ADJ_MAX
# MAGIC GROUP BY a.ORIGIN_STATION, a.ORIGIN_UTC
# MAGIC 
# MAGIC SELECT a.DEST_STATION, a.ORIGIN_UTC, max(wd.DATE) as DEST_MAX_DATE
# MAGIC FROM airlines as a, weather_metrics_imputed as wd
# MAGIC WHERE wd.STATION == a.DEST_STATION AND wd.DATE BETWEEN a.ORIGIN_UTC_ADJ_MIN AND a.ORIGIN_UTC_ADJ_MAX
# MAGIC GROUP BY a.DEST_STATION, a.ORIGIN_UTC
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC In the above SQL statements, we perform range queries to find the date timestamp of the weather observation for each airport station ID plus flight departure time to the imputed weather metrics record that is closest between 2 days and 2 hours prior to departure time. We use inner joins to combine the airlines dataset with the imputed weather metrics dataset and use the `BETWEEN` SQL command to filter using the specified inclusive range of timestamps.
# MAGIC 
# MAGIC > Add the `ORIGIN_MAX_DATE` and `DEST_MAX_DATE` columns to the airlines table
# MAGIC 
# MAGIC ```
# MAGIC airlines_final = airlines.join(weather_origin_helper, [airlines.ORIGIN_STATION == weather_origin_helper.ORIGIN_STATION, airlines.ORIGIN_UTC == weather_origin_helper.ORIGIN_UTC], 'left').select(*airlines, weather_origin_helper.ORIGIN_MAX_DATE)
# MAGIC 
# MAGIC airlines_final = airlines_final.join(weather_dest_helper, [airlines_final.DEST_STATION == weather_dest_helper.DEST_STATION, airlines_final.ORIGIN_UTC == weather_dest_helper.ORIGIN_UTC], 'left').select(*airlines_final, weather_dest_helper.DEST_MAX_DATE)
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC In the above queries, we simply do left joins between the airlines dataset and the helper tables from above with the join condition of matching the station IDs and departure times to select all columns from the airlines dataset and include the max date columns. We perform left joins here because we want to ensure that we do not remove any data from our airlines dataset and those records that do not have any matches to weather observations within the specified range will simply be imputed in following queries prior to train / vailidation / test split.
# MAGIC 
# MAGIC > Perform the final data join query to combine the updated airlines dataset with the imputed weather metrics dataset
# MAGIC 
# MAGIC ```
# MAGIC SELECT a.*, avg(wmo.AVG_WND_SPEED) as AVG_WND_SPEED_ORIGIN, avg(wmo.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        min(wmo.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_ORIGIN, avg(wmo.AVG_VIS_DIS) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        min(wmo.MIN_VIS_DIS) as MIN_VIS_DIS_ORIGIN, avg(wmo.AVG_TMP_DEG) as AVG_TMP_DEG_ORIGIN,   
# MAGIC        avg(wmo.AVG_DEW_DEG) as AVG_DEW_DEG_ORIGIN, avg(wmo.AVG_SLP) as AVG_SLP_ORIGIN, 
# MAGIC        avg(wmd.AVG_WND_SPEED) as AVG_WND_SPEED_DEST, avg(wmd.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        min(wmd.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_DEST, avg(wmd.AVG_VIS_DIS) as AVG_VIS_DIS_DEST, 
# MAGIC        min(wmd.MIN_VIS_DIS) as MIN_VIS_DIS_DEST, avg(wmd.AVG_TMP_DEG) as AVG_TMP_DEG_DEST, 
# MAGIC        avg(wmd.AVG_DEW_DEG) as AVG_DEW_DEG_DEST, avg(wmd.AVG_SLP) as AVG_SLP_DEST
# MAGIC FROM airlines_final as a 
# MAGIC   LEFT JOIN weather_metrics_imputed as wmo 
# MAGIC     ON a.ORIGIN_STATION == wmo.STATION AND wmo.DATE == a.ORIGIN_MAX_DATE
# MAGIC   LEFT JOIN weather_metrics_imputed as wmd 
# MAGIC     ON a.DEST_STATION == wmd.STATION AND wmd.DATE == a.DEST_MAX_DATE
# MAGIC GROUP BY a.YEAR, a.QUARTER, a.MONTH, a.DAY_OF_MONTH, a.DAY_OF_WEEK, a.FL_DATE, a.OP_UNIQUE_CARRIER,
# MAGIC          a.OP_CARRIER_AIRLINE_ID, a.OP_CARRIER, a.TAIL_NUM, a.OP_CARRIER_FL_NUM, a.ORIGIN_AIRPORT_ID, 
# MAGIC          a.ORIGIN_AIRPORT_SEQ_ID, a.ORIGIN_CITY_MARKET_ID, a.ORIGIN, a.ORIGIN_CITY_NAME, a.ORIGIN_STATE_ABR,
# MAGIC          a.ORIGIN_STATE_FIPS, a.ORIGIN_STATE_NM, a.ORIGIN_WAC, a.DEST_AIRPORT_ID, a.DEST_AIRPORT_SEQ_ID,
# MAGIC          a.DEST_CITY_MARKET_ID, a.DEST, a.DEST_CITY_NAME, a.DEST_STATE_ABR, a.DEST_STATE_FIPS, a.DEST_STATE_NM, 
# MAGIC          a.DEST_WAC, a.CRS_DEP_TIME, a.DEP_TIME, a.DEP_DELAY, a.DEP_DELAY_NEW, a.DEP_DEL15, a.DEP_DELAY_GROUP,
# MAGIC          a.DEP_TIME_BLK, a.TAXI_OUT, a.WHEELS_OFF, a.WHEELS_ON, a.TAXI_IN, a.CRS_ARR_TIME, a.ARR_TIME,
# MAGIC          a.ARR_DELAY, a.ARR_DELAY_NEW, a.ARR_DEL15, a.ARR_DELAY_GROUP, a.ARR_TIME_BLK, a.CANCELLED, a.DIVERTED,
# MAGIC          a.CRS_ELAPSED_TIME, a.ACTUAL_ELAPSED_TIME, a.AIR_TIME, a.FLIGHTS, a.DISTANCE, a.DISTANCE_GROUP,
# MAGIC          a.DIV_AIRPORT_LANDINGS, a.ORIGIN_TZ, a.DEST_TZ, a.DEP_MIN, a.DEP_HOUR, a.ARR_MIN, a.ARR_HOUR, a.ORIGIN_TS,
# MAGIC          a.ORIGIN_UTC, a.DEST_TS, a.DEST_UTC, a.ORIGIN_FLIGHT_COUNT, a.DEST_FLIGHT_COUNT, a.ORIGIN_STATION, 
# MAGIC          a.ORIGIN_STATION_NAME, a.PAGERANK, a.DEST_STATION, a.DEST_STATION_NAME, a.ORIGIN_UTC_ADJ_MIN,
# MAGIC          a.ORIGIN_UTC_ADJ_MAX, a.ORIGIN_MAX_DATE, a.DEST_MAX_DATE
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC In the above query, we perform left joins with the updated airlines table and the imputed weather metrics table (for both origin and destination stations). Our join conditions are that the origin station and destination station IDs match with the IDs within the imputed weather metrics dataset and that the weather dates match with the previously found `ORIGIN_MAX_DATE` and `DEST_MAX_DATE` columns.
# MAGIC 
# MAGIC We note that in using our weather dataset, we did not specifically filter out any specific report types for observations. As a result, we may weather observations from the same date timestamp but with different report types. Therefore in our final data join query above, we have aggregated the average/min weather metric columns in order to create only one observation for each flight record to avoid the creation of additional rows in our final dataset for weather observations with different report types.
# MAGIC 
# MAGIC Total count: 31,171,199
# MAGIC 
# MAGIC This total count of our final dataset matches the size of the cleaned airlines dataset. Moving forward, we filter out the 2019 data from this table.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Repeat above steps in the Joins Procedure for the updated weather dataset for just 2019 data
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Due to the issue noticed for 2019 weather observations as described earlier in this notebook, we will repeat the above steps to recreate an imputed weather dataset for 2019 data. We will not create an imputed weather metrics (average / min measurements) dataset for 2019 and instead simply join the airlines dataset with the imputed weather dataset. More specifically, we do this because it is important that we retain the raw data measurements for our test data.
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/1551285172210152/command/1551285172210270)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Adding New Features From Feature Engineering
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275129334/command/4061768275130867)

# COMMAND ----------

data = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/strategy/data_final")

data.registerTempTable('data')

data_15_to_18 = spark.sql("""
SELECT *
FROM data
WHERE EXTRACT(YEAR FROM ORIGIN_UTC) < 2019
""")

display(data_15_to_18)
data_15_to_18.count()

# COMMAND ----------

data_19 = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/strategy/data_final_19")

display(data_19)
data_19.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Below are all the new features we sequentially add to the above datasets.**
# MAGIC 
# MAGIC > ORIGIN_FLIGHT_COUNT, DEST_FLIGHT_COUNT
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC We perform left joins for each of the above datasets with the `flights_per_day` helper table (for origin and destination) 
# MAGIC - ORIGIN and DEST columns from above datasets match with the IATA column in the helper table 
# MAGIC - Flight departure time (ORIGIN_UTC) matches with the DATE column in the helper table
# MAGIC - Flight arrival time (DEST_UTC) matches with the DATE column in the helper table
# MAGIC 
# MAGIC > DELAYS_SO_FAR
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC We perform a left join for each of the above updated datasets with the `delays_so_far` helper table
# MAGIC - TAIL_NUM from above datasets matches with TAIL_NUM in helper table
# MAGIC - Flight departure time (ORIGIN_UC) matches with flight departure time (ORIGIN_UTC) in helper table
# MAGIC 
# MAGIC > WEST_TO_EAST, CRS_ELAPSED_TIME_AVG_DIFF
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC We perform inner joins for each of the above updated datasets with the `airlines_avg_time_west_east` helper table
# MAGIC - In order to join these datasets, we used the following combination of unique columns:
# MAGIC   - ORIGIN airport
# MAGIC   - TAIL_NUM
# MAGIC   - ORIGIN_UTC departure time
# MAGIC   - DEST airport
# MAGIC   - OP_CARRIER_FL_NUM flight number
# MAGIC 
# MAGIC > MINUTES_AFTER_MIDNIGHT_ORIGIN, MINUTES_AFTER_MIDNIGHT_DEST
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC No joins are necessary to add these features to the above updated datasets.
# MAGIC 
# MAGIC > HOLIDAY_WEEK
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC We perform left joins for each of the above updated datasets with the `holiday_week_beg_end` helper table
# MAGIC - We use a SQL `CASE WHEN` statement to depict if a flight departs during a holiday week (1) or not (0)
# MAGIC - Uses a range query to check if ORIGIN_UTC flight departure time is between beginning of the holiday week (HOLIDAY_WEEK_BEGIN) and end (HOLIDAY_WEEK_END)
# MAGIC 
# MAGIC > DEP_HOUR_BIN, ARR_HOUR_BIN
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC No joins are necessary to add these features to the above updated datasets.
# MAGIC 
# MAGIC > NETWORK_CONGESTION
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC We combine three helper tables for network congestion (train, validation, and test) into a single helper table. We then perform inner joins for each of the above updated datasets with the combined `network_congestion_all` helper table.
# MAGIC - ORIGIN_UTC_ADJ_MAX (timestamp for 2 hours prior to departure time) matches with ORIGIN_UTC_ADJ_MAX from helper table
# MAGIC 
# MAGIC > ORIGIN_PR, DEST_PR
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC We perform left joins for each of the above updated datasets with the `airport_meta` table
# MAGIC - ORIGIN airport matches with IATA code from `airport_meta`
# MAGIC - DEST airport matches with IATA code from `airport_meta`
# MAGIC 
# MAGIC > IS_MORNING_FLIGHT, IS_EVENING_FLIGHT
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC No joins are necessary to add these features to the above updated datasets.
# MAGIC - We use SQL `CASE WHEN` statements to depict if the flight is a morning flight (DEP_HOUR_BIN == 1) or evening flight (DEP_HOUR_BIN == 3)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Train / Validation / Test Split
# MAGIC 
# MAGIC Detailed section [here](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275129334/command/4061768275131010)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Train

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Split Weather Imputed Table To Use 2015 - 2017 Data
# MAGIC 
# MAGIC ```
# MAGIC SELECT *
# MAGIC FROM weather_imputed
# MAGIC WHERE YEAR < 2018
# MAGIC ```
# MAGIC 
# MAGIC > Create Helper Table For Average Metrics Grouped By Stations Using Train Data
# MAGIC 
# MAGIC ```
# MAGIC SELECT STATION, avg(WND_SPEED) as AVG_WND_SPEED, avg(CIG_HEIGHT) as AVG_CIG_HEIGHT, min(CIG_HEIGHT) as MIN_CIG_HEIGHT,
# MAGIC        avg(VIS_DIS) as AVG_VIS_DIS, min(VIS_DIS) as MIN_VIS_DIS, avg(TMP_DEGREE) as AVG_TMP_DEG, avg(DEW_DEGREE) as AVG_DEW_DEG,
# MAGIC        avg(SLP_PRESSURE) as AVG_SLP
# MAGIC FROM train_weather_imputed
# MAGIC GROUP BY STATION
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Compute average/min weather metrics for each unique station ID to use for phase 2 imputation.

# COMMAND ----------

station_average_helper = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/project_data/helpers/station_average_helper")
display(station_average_helper)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Filter `data_15_to_18` Dataset To Just Use Train Data Years (2015 - 2017) - train_data
# MAGIC 
# MAGIC ```
# MAGIC SELECT *
# MAGIC FROM data_15_to_18
# MAGIC WHERE EXTRACT(YEAR FROM ORIGIN_UTC) < 2018
# MAGIC ```
# MAGIC 
# MAGIC > Impute Weather Measurement Nulls With Station Average Helper Table (Phase-2 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT,
# MAGIC        coalesce(t.AVG_WND_SPEED_ORIGIN, saho.AVG_WND_SPEED) as AVG_WND_SPEED_ORIGIN, coalesce(t.AVG_CIG_HEIGHT_ORIGIN, saho.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_ORIGIN, saho.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_ORIGIN, coalesce(t.AVG_VIS_DIS_ORIGIN, saho.AVG_VIS_DIS) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        coalesce(t.MIN_VIS_DIS_ORIGIN, saho.MIN_VIS_DIS) as MIN_VIS_DIS_ORIGIN, coalesce(t.AVG_TMP_DEG_ORIGIN, saho.AVG_TMP_DEG) as AVG_TMP_DEG_ORIGIN, 
# MAGIC        coalesce(t.AVG_DEW_DEG_ORIGIN, saho.AVG_DEW_DEG) as AVG_DEW_DEG_ORIGIN, coalesce(t.AVG_SLP_ORIGIN, saho.AVG_SLP) as AVG_SLP_ORIGIN, 
# MAGIC        coalesce(t.AVG_WND_SPEED_DEST, sahd.AVG_WND_SPEED) as AVG_WND_SPEED_DEST, coalesce(t.AVG_CIG_HEIGHT_DEST, sahd.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_DEST, sahd.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_DEST, coalesce(t.AVG_VIS_DIS_DEST, sahd.AVG_VIS_DIS) as AVG_VIS_DIS_DEST, coalesce(t.MIN_VIS_DIS_DEST, sahd.MIN_VIS_DIS) as MIN_VIS_DIS_DEST, 
# MAGIC        coalesce(t.AVG_TMP_DEG_DEST, sahd.AVG_TMP_DEG) as AVG_TMP_DEG_DEST, coalesce(t.AVG_DEW_DEG_DEST, sahd.AVG_DEW_DEG) as AVG_DEW_DEG_DEST, coalesce(t.AVG_SLP_DEST, sahd.AVG_SLP) as AVG_SLP_DEST
# MAGIC FROM train_data as t 
# MAGIC   LEFT JOIN station_average_helper as saho
# MAGIC     ON t.ORIGIN_STATION == saho.STATION
# MAGIC   LEFT JOIN station_average_helper as sahd
# MAGIC     ON t.DEST_STATION == sahd.STATION
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Perform left joins with the train_data and station average helper table (for origin and destination) matching the origin and destination station IDs for each table. Impute null values for the matching origin and destination stations using the SQL `coalesce` function to use values from the helper table.
# MAGIC 
# MAGIC > Create Helper Table To Store Train Weather Metrics Averages (Column-wise) (Phase-3 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT avg(AVG_WND_SPEED_ORIGIN) as AVG_WND_SPEED_ORIGIN, avg(AVG_CIG_HEIGHT_ORIGIN) as AVG_CIG_HEIGHT_ORIGIN, avg(MIN_CIG_HEIGHT_ORIGIN) as MIN_CIG_HEIGHT_ORIGIN, 
# MAGIC        avg(AVG_VIS_DIS_ORIGIN) as AVG_VIS_DIS_ORIGIN, avg(MIN_VIS_DIS_ORIGIN) as MIN_VIS_DIS_ORIGIN, avg(AVG_TMP_DEG_ORIGIN) as AVG_TMP_DEG_ORIGIN, avg(AVG_DEW_DEG_ORIGIN) as AVG_DEW_DEG_ORIGIN, 
# MAGIC        avg(AVG_SLP_ORIGIN) as AVG_SLP_ORIGIN, avg(AVG_WND_SPEED_DEST) as AVG_WND_SPEED_DEST, avg(AVG_CIG_HEIGHT_DEST) as AVG_CIG_HEIGHT_DEST, avg(MIN_CIG_HEIGHT_DEST) as MIN_CIG_HEIGHT_DEST, 
# MAGIC        avg(AVG_VIS_DIS_DEST) as AVG_VIS_DIS_DEST, avg(MIN_VIS_DIS_DEST) as MIN_VIS_DIS_DEST, avg(AVG_TMP_DEG_DEST) as AVG_TMP_DEG_DEST, avg(AVG_DEW_DEG_DEST) as AVG_DEW_DEG_DEST, 
# MAGIC        avg(AVG_SLP_DEST) as AVG_SLP_DEST
# MAGIC FROM train_avg_imp
# MAGIC ```

# COMMAND ----------

train_averages = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/project_data/helpers/train_averages")
display(train_averages)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Perform Column Imputation On Train Data Using Above Helper Table (Phase-3 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT,
# MAGIC        coalesce(t.AVG_WND_SPEED_ORIGIN, ta.AVG_WND_SPEED_ORIGIN) as AVG_WND_SPEED_ORIGIN, coalesce(t.AVG_CIG_HEIGHT_ORIGIN, ta.AVG_CIG_HEIGHT_ORIGIN) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_ORIGIN, ta.MIN_CIG_HEIGHT_ORIGIN) as MIN_CIG_HEIGHT_ORIGIN, coalesce(t.AVG_VIS_DIS_ORIGIN, ta.AVG_VIS_DIS_ORIGIN) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        coalesce(t.MIN_VIS_DIS_ORIGIN, ta.MIN_VIS_DIS_ORIGIN) as MIN_VIS_DIS_ORIGIN, coalesce(t.AVG_TMP_DEG_ORIGIN, ta.AVG_TMP_DEG_ORIGIN) as AVG_TMP_DEG_ORIGIN, 
# MAGIC        coalesce(t.AVG_DEW_DEG_ORIGIN, ta.AVG_DEW_DEG_ORIGIN) as AVG_DEW_DEG_ORIGIN, coalesce(t.AVG_SLP_ORIGIN, ta.AVG_SLP_ORIGIN) as AVG_SLP_ORIGIN, 
# MAGIC        coalesce(t.AVG_WND_SPEED_DEST, ta.AVG_WND_SPEED_DEST) as AVG_WND_SPEED_DEST, coalesce(t.AVG_CIG_HEIGHT_DEST, ta.AVG_CIG_HEIGHT_DEST) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_DEST, ta.MIN_CIG_HEIGHT_DEST) as MIN_CIG_HEIGHT_DEST, coalesce(t.AVG_VIS_DIS_DEST, ta.AVG_VIS_DIS_DEST) as AVG_VIS_DIS_DEST, 
# MAGIC        coalesce(t.MIN_VIS_DIS_DEST, ta.MIN_VIS_DIS_DEST) as MIN_VIS_DIS_DEST, 
# MAGIC        coalesce(t.AVG_TMP_DEG_DEST, ta.AVG_TMP_DEG_DEST) as AVG_TMP_DEG_DEST, coalesce(t.AVG_DEW_DEG_DEST, ta.AVG_DEW_DEG_DEST) as AVG_DEW_DEG_DEST, coalesce(t.AVG_SLP_DEST, ta.AVG_SLP_DEST) as AVG_SLP_DEST
# MAGIC FROM train_avg_imp as t, train_averages as ta
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Perform an inner join with the updated train data and the above `train_averages` helper table to simply impute all remaining null values for the weather measurements column-wise. Again, we use the SQL `coalesce` function to do so.
# MAGIC 
# MAGIC At this point, our train data is null-free.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Validation

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Filter `data_15_to_18` dataset to create the validation data
# MAGIC 
# MAGIC ```
# MAGIC SELECT *
# MAGIC FROM data_15_to_18
# MAGIC WHERE EXTRACT(YEAR FROM ORIGIN_UTC) == 2018
# MAGIC ```
# MAGIC 
# MAGIC > Currently, the `data_15_to_18` uses the phase-1 imputed average/min weather metrics data for its measurements. Since we want to treat our validation data just like our test data, we will replace these weather metrics with the imputed weather data (validation_weather_imputed)
# MAGIC 
# MAGIC ```
# MAGIC SELECT *
# MAGIC FROM weather_imputed
# MAGIC WHERE YEAR == 2018
# MAGIC ```
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT, avg(wmo.WND_SPEED) as AVG_WND_SPEED_ORIGIN, avg(wmo.CIG_HEIGHT) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        min(wmo.CIG_HEIGHT) as MIN_CIG_HEIGHT_ORIGIN, avg(wmo.VIS_DIS) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        min(wmo.VIS_DIS) as MIN_VIS_DIS_ORIGIN, avg(wmo.TMP_DEGREE) as AVG_TMP_DEG_ORIGIN,   
# MAGIC        avg(wmo.DEW_DEGREE) as AVG_DEW_DEG_ORIGIN, avg(wmo.SLP_PRESSURE) as AVG_SLP_ORIGIN, 
# MAGIC        avg(wmd.WND_SPEED) as AVG_WND_SPEED_DEST, avg(wmd.CIG_HEIGHT) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        min(wmd.CIG_HEIGHT) as MIN_CIG_HEIGHT_DEST, avg(wmd.VIS_DIS) as AVG_VIS_DIS_DEST, 
# MAGIC        min(wmd.VIS_DIS) as MIN_VIS_DIS_DEST, avg(wmd.TMP_DEGREE) as AVG_TMP_DEG_DEST, 
# MAGIC        avg(wmd.DEW_DEGREE) as AVG_DEW_DEG_DEST, avg(wmd.SLP_PRESSURE) as AVG_SLP_DEST
# MAGIC FROM validation_data as t
# MAGIC   LEFT JOIN validation_weather_imputed as wmo 
# MAGIC     ON t.ORIGIN_STATION == wmo.STATION AND wmo.DATE == t.ORIGIN_MAX_DATE
# MAGIC   LEFT JOIN validation_weather_imputed as wmd 
# MAGIC     ON t.DEST_STATION == wmd.STATION AND wmd.DATE == t.DEST_MAX_DATE
# MAGIC WHERE EXTRACT(YEAR FROM t.ORIGIN_UTC) == 2018
# MAGIC GROUP BY t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT
# MAGIC ```
# MAGIC 
# MAGIC > Impute Validation Weather Measurements Using Station Average Helper Table (Created From Train Data) (Phase-2 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT,
# MAGIC        coalesce(t.AVG_WND_SPEED_ORIGIN, saho.AVG_WND_SPEED) as AVG_WND_SPEED_ORIGIN, coalesce(t.AVG_CIG_HEIGHT_ORIGIN, saho.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_ORIGIN, saho.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_ORIGIN, coalesce(t.AVG_VIS_DIS_ORIGIN, saho.AVG_VIS_DIS) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        coalesce(t.MIN_VIS_DIS_ORIGIN, saho.MIN_VIS_DIS) as MIN_VIS_DIS_ORIGIN, coalesce(t.AVG_TMP_DEG_ORIGIN, saho.AVG_TMP_DEG) as AVG_TMP_DEG_ORIGIN, 
# MAGIC        coalesce(t.AVG_DEW_DEG_ORIGIN, saho.AVG_DEW_DEG) as AVG_DEW_DEG_ORIGIN, coalesce(t.AVG_SLP_ORIGIN, saho.AVG_SLP) as AVG_SLP_ORIGIN, 
# MAGIC        coalesce(t.AVG_WND_SPEED_DEST, sahd.AVG_WND_SPEED) as AVG_WND_SPEED_DEST, coalesce(t.AVG_CIG_HEIGHT_DEST, sahd.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_DEST, sahd.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_DEST, coalesce(t.AVG_VIS_DIS_DEST, sahd.AVG_VIS_DIS) as AVG_VIS_DIS_DEST, coalesce(t.MIN_VIS_DIS_DEST, sahd.MIN_VIS_DIS) as MIN_VIS_DIS_DEST, 
# MAGIC        coalesce(t.AVG_TMP_DEG_DEST, sahd.AVG_TMP_DEG) as AVG_TMP_DEG_DEST, coalesce(t.AVG_DEW_DEG_DEST, sahd.AVG_DEW_DEG) as AVG_DEW_DEG_DEST, coalesce(t.AVG_SLP_DEST, sahd.AVG_SLP) as AVG_SLP_DEST
# MAGIC FROM validation_data as t 
# MAGIC   LEFT JOIN station_average_helper as saho
# MAGIC     ON t.ORIGIN_STATION == saho.STATION
# MAGIC   LEFT JOIN station_average_helper as sahd
# MAGIC     ON t.DEST_STATION == sahd.STATION
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Perform left joins with the validation_data and station average helper table (for origin and destination) matching the origin and destination station IDs for each table. Impute null values for the matching origin and destination stations using the SQL `coalesce` function to use values from the helper table.
# MAGIC 
# MAGIC > Perform Column Imputation On Validation Data Using Train Averages Helper Table (Phase-3 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT,
# MAGIC        coalesce(t.AVG_WND_SPEED_ORIGIN, ta.AVG_WND_SPEED_ORIGIN) as AVG_WND_SPEED_ORIGIN, coalesce(t.AVG_CIG_HEIGHT_ORIGIN, ta.AVG_CIG_HEIGHT_ORIGIN) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_ORIGIN, ta.MIN_CIG_HEIGHT_ORIGIN) as MIN_CIG_HEIGHT_ORIGIN, coalesce(t.AVG_VIS_DIS_ORIGIN, ta.AVG_VIS_DIS_ORIGIN) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        coalesce(t.MIN_VIS_DIS_ORIGIN, ta.MIN_VIS_DIS_ORIGIN) as MIN_VIS_DIS_ORIGIN, coalesce(t.AVG_TMP_DEG_ORIGIN, ta.AVG_TMP_DEG_ORIGIN) as AVG_TMP_DEG_ORIGIN, 
# MAGIC        coalesce(t.AVG_DEW_DEG_ORIGIN, ta.AVG_DEW_DEG_ORIGIN) as AVG_DEW_DEG_ORIGIN, coalesce(t.AVG_SLP_ORIGIN, ta.AVG_SLP_ORIGIN) as AVG_SLP_ORIGIN, 
# MAGIC        coalesce(t.AVG_WND_SPEED_DEST, ta.AVG_WND_SPEED_DEST) as AVG_WND_SPEED_DEST, coalesce(t.AVG_CIG_HEIGHT_DEST, ta.AVG_CIG_HEIGHT_DEST) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_DEST, ta.MIN_CIG_HEIGHT_DEST) as MIN_CIG_HEIGHT_DEST, coalesce(t.AVG_VIS_DIS_DEST, ta.AVG_VIS_DIS_DEST) as AVG_VIS_DIS_DEST, 
# MAGIC        coalesce(t.MIN_VIS_DIS_DEST, ta.MIN_VIS_DIS_DEST) as MIN_VIS_DIS_DEST, 
# MAGIC        coalesce(t.AVG_TMP_DEG_DEST, ta.AVG_TMP_DEG_DEST) as AVG_TMP_DEG_DEST, coalesce(t.AVG_DEW_DEG_DEST, ta.AVG_DEW_DEG_DEST) as AVG_DEW_DEG_DEST, coalesce(t.AVG_SLP_DEST, ta.AVG_SLP_DEST) as AVG_SLP_DEST
# MAGIC FROM validation_avg_imp as t, train_averages as ta
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Perform an inner join with the updated validation data and the above `train_averages` helper table to simply impute all remaining null values for the weather measurements column-wise. Again, we use the SQL `coalesce` function to do so.
# MAGIC 
# MAGIC At this point, our validation data is null-free.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC > Load Test Data With Newly Added Features
# MAGIC 
# MAGIC ```
# MAGIC test_data = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/team20SSDK/project_data/data_19")
# MAGIC ```
# MAGIC 
# MAGIC > Impute Test Weather Metrics Using Station Average Helper Table (Create From Train Data) (Phase-2 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT,
# MAGIC        coalesce(t.AVG_WND_SPEED_ORIGIN, saho.AVG_WND_SPEED) as AVG_WND_SPEED_ORIGIN, coalesce(t.AVG_CIG_HEIGHT_ORIGIN, saho.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_ORIGIN, saho.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_ORIGIN, coalesce(t.AVG_VIS_DIS_ORIGIN, saho.AVG_VIS_DIS) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        coalesce(t.MIN_VIS_DIS_ORIGIN, saho.MIN_VIS_DIS) as MIN_VIS_DIS_ORIGIN, coalesce(t.AVG_TMP_DEG_ORIGIN, saho.AVG_TMP_DEG) as AVG_TMP_DEG_ORIGIN, 
# MAGIC        coalesce(t.AVG_DEW_DEG_ORIGIN, saho.AVG_DEW_DEG) as AVG_DEW_DEG_ORIGIN, coalesce(t.AVG_SLP_ORIGIN, saho.AVG_SLP) as AVG_SLP_ORIGIN, 
# MAGIC        coalesce(t.AVG_WND_SPEED_DEST, sahd.AVG_WND_SPEED) as AVG_WND_SPEED_DEST, coalesce(t.AVG_CIG_HEIGHT_DEST, sahd.AVG_CIG_HEIGHT) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_DEST, sahd.MIN_CIG_HEIGHT) as MIN_CIG_HEIGHT_DEST, coalesce(t.AVG_VIS_DIS_DEST, sahd.AVG_VIS_DIS) as AVG_VIS_DIS_DEST, coalesce(t.MIN_VIS_DIS_DEST, sahd.MIN_VIS_DIS) as MIN_VIS_DIS_DEST, 
# MAGIC        coalesce(t.AVG_TMP_DEG_DEST, sahd.AVG_TMP_DEG) as AVG_TMP_DEG_DEST, coalesce(t.AVG_DEW_DEG_DEST, sahd.AVG_DEW_DEG) as AVG_DEW_DEG_DEST, coalesce(t.AVG_SLP_DEST, sahd.AVG_SLP) as AVG_SLP_DEST
# MAGIC FROM test_data as t 
# MAGIC   LEFT JOIN station_average_helper as saho
# MAGIC     ON t.ORIGIN_STATION == saho.STATION
# MAGIC   LEFT JOIN station_average_helper as sahd
# MAGIC     ON t.DEST_STATION == sahd.STATION
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Perform left joins with the test_data and station average helper table (for origin and destination) matching the origin and destination station IDs for each table. Impute null values for the matching origin and destination stations using the SQL `coalesce` function to use values from the helper table.
# MAGIC 
# MAGIC > Perform Column Imputation On Test Data Using Train Averages Helper Table (Phase-3 Imputation)
# MAGIC 
# MAGIC ```
# MAGIC SELECT t.YEAR, t.QUARTER, t.MONTH, t.DAY_OF_MONTH, t.DAY_OF_WEEK, t.FL_DATE, t.OP_UNIQUE_CARRIER, t.OP_CARRIER_AIRLINE_ID, t.OP_CARRIER, t.TAIL_NUM, t.OP_CARRIER_FL_NUM, t.ORIGIN_AIRPORT_ID, t.ORIGIN_AIRPORT_SEQ_ID,
# MAGIC        t.ORIGIN_CITY_MARKET_ID, t.ORIGIN, t.ORIGIN_CITY_NAME, t.ORIGIN_STATE_ABR, t.ORIGIN_STATE_FIPS, t.ORIGIN_STATE_NM, t.ORIGIN_WAC, t.DEST_AIRPORT_ID, t.DEST_AIRPORT_SEQ_ID, t.DEST_CITY_MARKET_ID, t.DEST,
# MAGIC        t.DEST_CITY_NAME, t.DEST_STATE_ABR, t.DEST_STATE_FIPS, t.DEST_STATE_NM, t.DEST_WAC, t.CRS_DEP_TIME, t.DEP_TIME, t.DEP_DELAY, t.DEP_DELAY_NEW, t.DEP_DEL15, t.DEP_DELAY_GROUP, t.DEP_TIME_BLK, t.TAXI_OUT,
# MAGIC        t.WHEELS_OFF, t.WHEELS_ON, t.TAXI_IN, t.CRS_ARR_TIME, t.ARR_TIME, t.ARR_DELAY, t.ARR_DELAY_NEW, t.ARR_DEL15, t.ARR_DELAY_GROUP, t.ARR_TIME_BLK, t.CANCELLED, t.DIVERTED, t.CRS_ELAPSED_TIME, t.ACTUAL_ELAPSED_TIME,
# MAGIC        t.AIR_TIME, t.FLIGHTS, t.DISTANCE, t.DISTANCE_GROUP, t.DIV_AIRPORT_LANDINGS, t.ORIGIN_TZ, t.DEST_TZ, t.DEP_MIN, t.DEP_HOUR, t.ARR_MIN, t.ARR_HOUR, t.ORIGIN_TS, t.ORIGIN_UTC, t.DEST_TS, t.DEST_UTC, 
# MAGIC        t.ORIGIN_STATION, t.ORIGIN_STATION_NAME, t.DEST_STATION, t.DEST_STATION_NAME, t.ORIGIN_UTC_ADJ_MIN, t.ORIGIN_UTC_ADJ_MAX, t.ORIGIN_MAX_DATE, t.DEST_MAX_DATE, t.ORIGIN_FLIGHT_COUNT, t.DEST_FLIGHT_COUNT, 
# MAGIC        t.DELAYS_SO_FAR, t.CRS_ELAPSED_TIME_AVG_DIFF, t.WEST_TO_EAST, t.MINUTES_AFTER_MIDNIGHT_ORIGIN, t.MINUTES_AFTER_MIDNIGHT_DEST, t.HOLIDAY_WEEK, t.DEP_HOUR_BIN, t.ARR_HOUR_BIN, t.NETWORK_CONGESTION, 
# MAGIC        t.ORIGIN_PR, t.DEST_PR, t.IS_MORNING_FLIGHT, t.IS_EVENING_FLIGHT,
# MAGIC        coalesce(t.AVG_WND_SPEED_ORIGIN, ta.AVG_WND_SPEED_ORIGIN) as AVG_WND_SPEED_ORIGIN, coalesce(t.AVG_CIG_HEIGHT_ORIGIN, ta.AVG_CIG_HEIGHT_ORIGIN) as AVG_CIG_HEIGHT_ORIGIN, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_ORIGIN, ta.MIN_CIG_HEIGHT_ORIGIN) as MIN_CIG_HEIGHT_ORIGIN, coalesce(t.AVG_VIS_DIS_ORIGIN, ta.AVG_VIS_DIS_ORIGIN) as AVG_VIS_DIS_ORIGIN, 
# MAGIC        coalesce(t.MIN_VIS_DIS_ORIGIN, ta.MIN_VIS_DIS_ORIGIN) as MIN_VIS_DIS_ORIGIN, coalesce(t.AVG_TMP_DEG_ORIGIN, ta.AVG_TMP_DEG_ORIGIN) as AVG_TMP_DEG_ORIGIN, 
# MAGIC        coalesce(t.AVG_DEW_DEG_ORIGIN, ta.AVG_DEW_DEG_ORIGIN) as AVG_DEW_DEG_ORIGIN, coalesce(t.AVG_SLP_ORIGIN, ta.AVG_SLP_ORIGIN) as AVG_SLP_ORIGIN, 
# MAGIC        coalesce(t.AVG_WND_SPEED_DEST, ta.AVG_WND_SPEED_DEST) as AVG_WND_SPEED_DEST, coalesce(t.AVG_CIG_HEIGHT_DEST, ta.AVG_CIG_HEIGHT_DEST) as AVG_CIG_HEIGHT_DEST, 
# MAGIC        coalesce(t.MIN_CIG_HEIGHT_DEST, ta.MIN_CIG_HEIGHT_DEST) as MIN_CIG_HEIGHT_DEST, coalesce(t.AVG_VIS_DIS_DEST, ta.AVG_VIS_DIS_DEST) as AVG_VIS_DIS_DEST, 
# MAGIC        coalesce(t.MIN_VIS_DIS_DEST, ta.MIN_VIS_DIS_DEST) as MIN_VIS_DIS_DEST, 
# MAGIC        coalesce(t.AVG_TMP_DEG_DEST, ta.AVG_TMP_DEG_DEST) as AVG_TMP_DEG_DEST, coalesce(t.AVG_DEW_DEG_DEST, ta.AVG_DEW_DEG_DEST) as AVG_DEW_DEG_DEST, coalesce(t.AVG_SLP_DEST, ta.AVG_SLP_DEST) as AVG_SLP_DEST
# MAGIC FROM test_avg_imp as t, train_averages as ta
# MAGIC ```
# MAGIC 
# MAGIC **Description**
# MAGIC 
# MAGIC Perform an inner join with the updated test data and the above `train_averages` helper table to simply impute all remaining null values for the weather measurements column-wise. Again, we use the SQL `coalesce` function to do so.
# MAGIC 
# MAGIC At this point, our test data is null-free.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Toy Model
# MAGIC 
# MAGIC #### Decision Trees
# MAGIC Decisions trees are a classification tool which takes its name from the tree-like structure it forms where each node is a test on a feature and each branch divides the dataset on the outcome of that test.  We found that decision trees have a number of qualities which are particulary suited for our task.  A few of the most promenant are:
# MAGIC * They are interpertable since each node can be phrased as a question that tests one of the features.
# MAGIC * They naturally express feature importance.  Since we have hundreds of potential features, many of which may have undesirable qualities such as colinearity, this model provides an approach for sensible feature removal.
# MAGIC * They can are robust to unseen categorical values.  We found that over time different airports and weather station come online, and in larger time scales, the airlines themselves are reorganized as businesses, and any of these events can cause a change in the categorical values our model receives.
# MAGIC * They are resilant to missing values.  We also found that many of the weather measurements are inconsistent or unusable, and although we are putting significant effort into filling those values, it warrants forseeing that there may be intances of missing values that will still want to predict against in the future.
# MAGIC * Of course, since the volume of data we're training against is so large, the fact that random forests are constructed from indepented trees which can be built in parallel is an important performance consideration.
# MAGIC 
# MAGIC Our original baseline model was built using linear regression, which also has very nice qualities in regards to interpertability, idenification of feature performance and parallel implementation. However would also have to invest additional resources in preprocessing and creating deterministic outcomes for changes in categorical variables or unexpected missing variable that would potentially take away from better feature engineering and feature selection, as well as set us up for a less robust inference unit production.

# COMMAND ----------

# MAGIC %md
# MAGIC When we created our baseline linear regression model, we excluded a considerable minority of the data because of the amount of missing and unusable values in the raw weather data.  While we were still pursing several strategies for filling in these gaps with sensible values, we felt that it would be prudent to examine classification algrithms that would be robust to these missing values to cover our bases in case the weather imputation strategies ran into trouble.  Also, seeing these challenges in the development enviornments lead us to believe that these challenges may be even more severe in the production enviorment, so being resilant to these issues would be beneficial.  At the same time the linear regression model had many important qualities we didn't want to lose, such as explainability and a natural approach to feature reduction.  Decision trees stood out as an algorithm that has one of the best mix of the benefits and data resilience that seemed appropriate to the problem at hand.
# MAGIC 
# MAGIC Decision trees classify a dataset by find a value in one of the parameters that can be used to divide the entire set into two groups.  Then each group is subsequently split over and over again, until all the remaining groups are pure, meaning that all of the samples in that group have the same classification.  Determining the optimal split point is achieved by trying to make each group as pure as possible, and the measure of purity is the gini index.  It is deviation from a perfectly equal distribution, expressed as \\(Gini Index = 1-\sum_{i=1}^{n} p_i^2\\) where \\(p\\) is the probability of each class.
# MAGIC This approach can have a tendency to follow the training data too closely thought, which is overfitting, so we are adopting the Random Forests variant of Decisions Trees, which builds a variety of trees from different random subsets of the training data, a process known as bagging.  Additionally, the trees are trained on random subsets of the features, which is known as boosting. When a prediction is made, each of the trees is evaluated and a majority vote becomes the decision.
# MAGIC 
# MAGIC Below we have created a small example of how a Decision Tree would handle a sample of data with potientially missing weather values.  This produces a model that has high explainablity and an easy way to view feature importance so we will be able to prune are model for more parsimony.

# COMMAND ----------

from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint

column_names =  ['ORIGIN','DEST','DEP_HOUR','ARR_HOUR','AVG_TMP_DEG_ORIGIN', 'DEP_DEL15']
origin = {"ATL": 0.0, "ORD": 1.0}
dest =  {"ATL": 0.0, "ORD": 1.0}

labeledpoints = [
  
  LabeledPoint(0.0, [origin["ATL"], dest['ORD'], 8, 9, -4.390625]),
  LabeledPoint(0.0, [origin["ATL"], dest['ORD'], 8, 9, 151.23295454545453]),
  LabeledPoint(0.0, [origin["ATL"], dest['ORD'], 21, 22, None]),
  LabeledPoint(1.0, [origin["ATL"], dest['ORD'], 21, 22, 187.9034090909091]),
  LabeledPoint(0.0, [origin["ATL"], dest['ORD'], 13, 14, 252.6086956521739]),
  LabeledPoint(1.0, [origin["ATL"], dest['ORD'], 6, 7, 220.31944444444446]),
  LabeledPoint(0.0, [origin["ORD"], dest['ATL'], 18, 21, 314.2142857142857]),
  LabeledPoint(1.0, [origin["ORD"], dest['ATL'], 18, 21, 285.17083333333335]),
  LabeledPoint(1.0, [origin["ORD"], dest['ATL'], 9, 12, None]),
  LabeledPoint(0.0, [origin["ORD"], dest['ATL'], 20, 23, 266.3333333333333]),
  LabeledPoint(0.0, [origin["ORD"], dest['ATL'], 12, 15, 190.712]),
  LabeledPoint(0.0, [origin["ORD"], dest['ATL'], 6, 9, 21.32089552238806])
]

toy_training_data = sc.parallelize(labeledpoints)

toy_model = DecisionTree.trainClassifier(data=toy_training_data,
         numClasses=2,
         categoricalFeaturesInfo={0: 2, 1:7})


# COMMAND ----------

# DBTITLE 1,Toy decision tree visualized
displayHTML("<img src ='https://github.com/seancampos/w261-fp-images/raw/master/tree2.png'>") 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Modelling 
# MAGIC 
# MAGIC In this section we would like to describe our journey through modelling. We will briefly touch upon our baseline model, and then move to later stage models for which we will highlight various metrics, confusion matrixes, correlation matrixes, and feature selection.
# MAGIC 
# MAGIC In all of our models, we have taken the approach below to handle imbalance in classes.
# MAGIC 
# MAGIC ##### Handling Imbalance in the dataset
# MAGIC ###### Low Preicision in minority class  (Delay)
# MAGIC 
# MAGIC As we've seen our data is highly imbalanced this would result in a model which'll be more biased towards predicting the majority class (No Delay). This is beacuse the algorithm will not have enough data to learn the patterns present in the minority class (Delay).That is why there will be high misclassification errors for the minority class and hence a low precision.
# MAGIC 
# MAGIC ###### Workaround
# MAGIC - Modify the current training algorithm to take into account the skewed distribution of the classes by giving different weights to the majority and minority classes.
# MAGIC 
# MAGIC This difference in weights will influence the classification of the classes during the training phase.
# MAGIC The idea is to penalize the misclassification made by the minority class by setting a higher class weight and at the same time reducing weight for the majority class.
# MAGIC 
# MAGIC ###### Implementation in LR
# MAGIC This is implemented in LR by modifying the **cost function** as below:
# MAGIC 
# MAGIC  $$log loss = \frac{1}{N}\sum_{x=1}^N [- w_0(y_i*(log{(\hat y_i)})+w_1((1-y_i)(log{1-(\hat y_i)})) ]$$
# MAGIC  
# MAGIC  $$ w_0 = weight class 0 $$
# MAGIC  $$ w_1 = weight class 1 $$

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Baseline Model- Logistic Regression(LR)
# MAGIC 
# MAGIC Our baseline model was a simple logistic regression to proedict the binary class probability. We were interested to explore a linear approach and observe the co-efficients and their magnitudes.
# MAGIC 
# MAGIC 
# MAGIC * **Variations on the baseline model**: 
# MAGIC   - Approach 1: [Vanila LR model](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2127515060485759/command/667068395499476)
# MAGIC  
# MAGIC               F1 Score:  0.352822366322951
# MAGIC               precision    recall  f1-score   support
# MAGIC 
# MAGIC          0.0       0.88      0.60      0.71     25073
# MAGIC          1.0       0.25      0.62      0.35      5360
# MAGIC 
# MAGIC         accuracy                           0.60     30433
# MAGIC         macro avg       0.56      0.61      0.53     30433
# MAGIC         weighted avg    0.77      0.60      0.65     30433
# MAGIC 
# MAGIC   - Approach 2: [LR model with interaction terms](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/667068395499511/command/667068395499568)
# MAGIC     - Hypothesis - The individual weather elements might not impact a flight schedule, but their interaction could.
# MAGIC     - Implementation - Included interaction terms
# MAGIC       - Scaled down Temp and Speed values(since they were scaled by 10)
# MAGIC       - Normalized the feature vector by 0 mean and 1 Std. Deviation
# MAGIC   
# MAGIC   
# MAGIC               Accuracy Score:  0.4456018138205238
# MAGIC               F1 Score:  0.3365316555249705
# MAGIC               precision    recall  f1-score   support
# MAGIC 
# MAGIC          0.0       0.90      0.37      0.52     25073
# MAGIC          1.0       0.21      0.80      0.34      5360
# MAGIC 
# MAGIC          accuracy                           0.45     30433
# MAGIC          macro avg       0.55      0.58      0.43     30433
# MAGIC          weighted avg    0.78      0.45      0.49     30433
# MAGIC   
# MAGIC 
# MAGIC ** Observations from the Baseline Model ** :
# MAGIC - We were able to extract the [co-efficients](https://docs.google.com/spreadsheets/u/3/d/1G_MgoBPHqiXU5jUFCUUeCDLa8M-g_mdqVuwX4x9wkUw/edit#gid=0) to get an idea of the strengths of the coefficients.
# MAGIC - To our surprise the inclusion of interaction terms did not improve metrics.This could be due to the fact that:
# MAGIC   - Interaction terms might have added to the noise resulting in poor metrics.
# MAGIC 
# MAGIC **Optimizations on Baseline model**
# MAGIC We tried to optimize on the training time by :
# MAGIC - selecting only the rows that had 'valid' data(removing missing elements represented by 999 etc.)
# MAGIC - selected a subset of features by analyzing the collinearity between features.
# MAGIC - Analyzed the flights from the busiest airports first ('Chicago' and 'Atlanta') during Q1 of 2015.
# MAGIC 
# MAGIC This paved way towards more sophisticated ensemble learning methods -Random Forests. We belive that a tree based approach will imrprove the shortcomings of a linear model.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Random Forest
# MAGIC 
# MAGIC From various research, we had a hunch that tree based models is what would yield our best results. We went into this exploration mid way through our feature engineering cycle, therefore we had the chance to really play with various features sets and a minimal amount of hyper parameter tuning.
# MAGIC 
# MAGIC **Showcased below is a table that captures a majority of training metrics from our Random Forest experiments.**

# COMMAND ----------

# DBTITLE 1,Validation Set Metrics
val_perf = pd.read_csv(f'/dbfs/mnt/mids-w261/team20SSDK/reports/val_perf.csv')
val_perf

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC #### Phase 2
# MAGIC ##### Kitchen Sink
# MAGIC Some challenges we faced during this process, was the training run time. Our initial strategy was to take teh **"kitchen sink"** approach, where we threw every single one of our features at the Random Forest to see how it would perform, and what features it would rank highly based on feature importance. 
# MAGIC 
# MAGIC We used Spark ML's **Pipeline** interface to accomplish modelling tasks as we built out pipeline stages that performed **String Indexing**, **One Hot Encoding (OHE)** (categorical features), and **Feature Vector Assembling**. By the end of these pre-processing steps, after OHE we ended up with **918** columns that were used to fit the model on train data.
# MAGIC 
# MAGIC This process took nearly **11.5 hrs**, of which **9.04 hrs** was model fit time, **54 min** was train prediction time, and **26.22 min** was validation prediction time. We quickly learned that as we increased the `maxDepth` hyper paramter that our model compute time took exponentially longer. In order to solve this, what we did was play around with models holding `maxDepth` a constant until we knew that we had a feature set that we were happy with, from there we would the maxDepth.
# MAGIC 
# MAGIC We quickly learned with the cluster provided, running any sort of ParamGrid / CrossValidation for hyper parameter tuning was not going to work. Lastly, as we looked at our **confusion matrix** and our **feature importance** table, we realized that our model was giving high precedence to `weather` features, taking up 76% of the top 21 features. We felt that we had to ensure that the model wasn't building a train bias with weather, therefore we had to get clever with our **feature engineering**.
# MAGIC 
# MAGIC Detailed notebook [here]
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2158640876510655/command/2158640876510656

# COMMAND ----------

displayHTML('<img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/kitchValMet.png?token=AEALQT2YTZIK2VC5R5SZ3DS73RRZE height=300>')


# COMMAND ----------

# DBTITLE 1,Confusion Matrix - rd_model_2
displayHTML('<img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/kitchenVal.png?token=AEALQT5BVLGWLTFFSL26HMS73RQXU height=300>')

# COMMAND ----------

# DBTITLE 1,Feature Importance - rd_model_2
displayHTML('<img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/kitchValFeat.png?token=AEALQTZWAC5KYQCSVTD2CDK73RQQM height=400>')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC #### Phase 3
# MAGIC ##### The Wilderness
# MAGIC 
# MAGIC <em>Reading, trial, error, results....Reading, trial, error,results....</em>
# MAGIC 
# MAGIC **Feature Engineering Section** highlighted the various features we built to capture:
# MAGIC * Weather
# MAGIC * Time of Day
# MAGIC * Propagation Delay
# MAGIC * Airport Popularity
# MAGIC * Network Congestion
# MAGIC * Estimated Trip Durations.
# MAGIC 
# MAGIC Given there is a lot of rich details to unpack here, we have built this **[notebook]** to take you through the analysis of this modelling phase which resulted in **feature selection** to build a parsimonious and generalizable, final model.
# MAGIC 
# MAGIC 
# MAGIC [notebook]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275142374/command/4061768275144362

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Out of the 60+ features we considered in this phase of modelling, based off feature importance the following **19** features consistently showed up at the top
# MAGIC 
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/top19.png?token=AEALQT6F6TA5EOFRAPGMIV273J6O6>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC As was detailed in our model summaries [notebook], we do a final feature selection to minimize collinearity, which leaves us with the following features. 
# MAGIC 
# MAGIC In addition you will notice a feature called `IS_EARLY_MORNING_FLIGHT`, in accordance with our business case of optimizing **precision** we conducted an [analysis] on **False Positives** to try and figure out under what circumstances we predict positive incorrectly. Through this we learned that during early morning times our false positive rate grows to be almost 50% of total prediction outcomes associated to the feature `MINUTES_AFTER_MIDNIGHT_ORIGIN`.
# MAGIC 
# MAGIC We added this new feature on the fly during our final modelling phase with variants on the below features presented [here]
# MAGIC 
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/is_early_morning_flight.png?token=AEALQTZ3JN4ANGDG77BYIGC73RIRE width=900 height=400>
# MAGIC 
# MAGIC [notebook]: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275142374/command/4061768275145592
# MAGIC [analysis]: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2834511320022313/command/2834511320025772
# MAGIC [here]: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4061768275145382/command/4061768275145384

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Model 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Set
# MAGIC 
# MAGIC ```
# MAGIC final_num = ['DELAYS_SO_FAR', 'MINUTES_AFTER_MIDNIGHT_ORIGIN','MINUTES_AFTER_MIDNIGHT_DEST',
# MAGIC                  'NETWORK_CONGESTION','AVG_VIS_DIS_ORIGIN','DEST_PR','ORIGIN_PR', 'AVG_DEW_DEG_ORIGIN',
# MAGIC                  'CRS_ELAPSED_TIME', 'AVG_WND_SPEED_ORIGIN','AVG_WND_SPEED_DEST',            
# MAGIC             ]
# MAGIC 
# MAGIC final_cat = ['QUARTER', 'IS_EARLY_MORNING_FLIGHT', 'DEP_HOUR_BIN','ARR_HOUR_BIN']
# MAGIC ```
# MAGIC 
# MAGIC Below is the correlation matrix for all the numerical features.
# MAGIC 
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/finalFeat.png?token=AEALQT7GJI2DZ5BP7RAZR7K73VOFU>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Given the dataset's that we have built, with the goal of predicting `DEPARTURE_DEL15` based off our EDA, it is reasonable to say that year over year trends in the data stay fairly consistent. This year 2020 has by all means been an exception, for our purposes studying the past 4 years has provided a good amount of data to bring meaning to the model. From research we have done, if it was possible we would like to extend our window of data to look back on to 5-6 years.
# MAGIC 
# MAGIC Once we have this 5-6 year window, it would be a shifting window year over year; this provides a scalable solution because our data size would be failry constant. Since we would be only retraining the model once per year, optimizing the training time is **not** a priority. Even if training took 24-48 hours on this would be acceptable since this problem doesn't require real-time training.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline
# MAGIC 
# MAGIC Please view our final model pipeline [here]. This covers all steps that were listed below.
# MAGIC 
# MAGIC When designing our model pipeline we wanted to be very exhaustive in terms of computing and saving all metrics and artifacts of the model creating process. Our pipeline captures the following facets for every model we run.
# MAGIC 
# MAGIC 
# MAGIC 1. Data Pre Processing
# MAGIC   * One Hot Encoding
# MAGIC   * Feature Vector Assembler
# MAGIC   * Fit Pipeline
# MAGIC 
# MAGIC 2. Model Training (Fit)
# MAGIC 
# MAGIC 3. Model Predict (Transform)
# MAGIC   * Train Data
# MAGIC   * Validation Data
# MAGIC   
# MAGIC 4. Scoring
# MAGIC   * AOC_ROC
# MAGIC   * AOC_PR
# MAGIC   
# MAGIC 5. Reporting
# MAGIC   * F1_Score
# MAGIC   * Accuracy
# MAGIC   * Precision
# MAGIC   * Recall
# MAGIC   * Confusion Matrix
# MAGIC   
# MAGIC 6. Model Serving 
# MAGIC    * Databricks Model Registry
# MAGIC    * ML Flow Single Node Serving
# MAGIC    
# MAGIC Other:
# MAGIC * Compute Time Logging
# MAGIC * Artifact Logging
# MAGIC 
# MAGIC 
# MAGIC We used **ML Flow** as a tool to track each of our model runs as an **"Experiment"** logged within Databricks, this let us seamlessly revisit past runs and track our progress. In addition we saved all predicted outputs, and feature importance tables into DBFS so that we could go back and generate metric reports to analyze our model performance.
# MAGIC 
# MAGIC The key to this pipeline was making the flow as configurable and consistent as possible, this way we could easily extend the Pipeline to new data, and new models experiments.
# MAGIC 
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/experiments.png?token=AEALQT6XH2ARY7EANBQSBYK73PJCO>
# MAGIC 
# MAGIC [here]:https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/2834511320014363/command/2834511320014365

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Deployment
# MAGIC 
# MAGIC Though the model fitting process is an offline job, predicting flight delays is a very much so **a real-time task**. We would need to deploy our model on a web server, for example using a combination of **NGinx** and **Gunicorn** to load balance a **Flask** server, we can expose an HTTP endpoint accessible through an API interface. The arguments to this API would be the various features that we use to base the prediction, of course now the request's would be coming in sporadically compared to when we emulate this by predicting on the test set.
# MAGIC 
# MAGIC To handle requests that have incomplete data arguments we would have to have business logic to impute these incomplete data requests on the fly based on averages that we have computed from training data, and to handle the high traffic of requests we would have the load balancer on our model serving cluster.
# MAGIC 
# MAGIC Of course the use cases can truly vary, but assuming this system is to support an organization like the **Federal Aviation Administration** who handle **45,000** average daily flights, with **5,400** aircrafts in the sky at peak operational times, the flight prediction system needs to be ready and robust enough to handle high throughput.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **ML Flow Model Serve UI**
# MAGIC <img src=https://raw.githubusercontent.com/UCB-w261/f20-karthikrbabu/master/Assignments/Final%20Project/SlideVisuals/modelServe.png?token=AEALQTZJ4WRPH547XQMF65S73KSV4>
# MAGIC 
# MAGIC **Sample Request Format:**
# MAGIC ```
# MAGIC curl -u token:$DATABRICKS_API_TOKEN $MODEL_VERSION_URI \
# MAGIC   -H 'Content-Type: application/json; format=pandas-records' \
# MAGIC   -d '[
# MAGIC     {
# MAGIC       "DELAYS_SO_FAR": 1,
# MAGIC       "MINUTES_AFTER_MIDNIGHT_ORIGIN": 330,
# MAGIC       "MINUTES_AFTER_MIDNIGHT_DEST": 380,
# MAGIC       "NETWORK_CONGESTION": 4012,
# MAGIC       "AVG_VIS_DIS_ORIGIN": 15288.5,      
# MAGIC       "DEST_PR": 0.0078058,
# MAGIC       "ORIGIN_PR": 0.0012741,
# MAGIC       "AVG_DEW_DEG_ORIGIN": 156,
# MAGIC       "CRS_ELAPSED_TIME": 50,
# MAGIC       "AVG_WND_SPEED_ORIGIN": 26,
# MAGIC       "AVG_WND_SPEED_DEST": 34.75,
# MAGIC       "QUARTER": 1,
# MAGIC       "DEP_HOUR_BIN": 3,
# MAGIC       "ARR_HOUR_BIN": 3,
# MAGIC       "IS_EARLY_MORNING_FLIGHT": 0
# MAGIC     }
# MAGIC   ]'
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC **Response:**
# MAGIC ```
# MAGIC >> {'status': 'Success', 'data': {'pred': 1, 'probabilities':[0.27634272, 0.72365728]}}
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC We'd like to summarize what our model has achieved and the journey of this project.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance
# MAGIC 
# MAGIC Looking back to our business case, our goal is to err on the side of caution. We would rather predict a flight to be `ON-TIME` even if it is not, this translates to optimizing the metric of **on-time class precision**. Therefore balancing the real life use case vs. our data scientist view to achieve the highest accuracy was a battle.
# MAGIC 
# MAGIC Relative to our baseline model, we have **15%** improvement in **on-time class precision** to a final number of **37%**. Relative to our baseline model, we have a **~30%** improvement in **accuracy** to a final number of **73.5%**. 
# MAGIC 
# MAGIC 
# MAGIC The metrics on our final model for train and test data are highlighted below. In addition we share our final confusion matrices and feature importance rankings.

# COMMAND ----------

model_version = 'rf_model_final'
feat_imp = pd.read_csv(f'/dbfs/mnt/mids-w261/team20SSDK/models/model_meta/{model_version}/feat_imp.csv')
trainScoreAndLabels_pd = pd.read_csv(f'/dbfs/mnt/mids-w261/team20SSDK/models/model_meta/{model_version}/train_pred.csv')
testScoreAndLabels_pd = pd.read_csv(f'/dbfs/mnt/mids-w261/team20SSDK/models/model_meta/{model_version}/test_pred.csv')

y_train_true = trainScoreAndLabels_pd["label"]
y_train_pred = trainScoreAndLabels_pd["raw"]
conf_mat_train = confusion_matrix(y_train_true, y_train_pred)

print("Train Set:")
print("Accuracy Score: ", accuracy_score(y_train_true, y_train_pred))
print("F1 Score: ", f1_score(y_train_true, y_train_pred))
print(classification_report(y_train_true, y_train_pred))


df_cm_train = pd.DataFrame(conf_mat_train, range(2), range(2))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm_train, annot=True, annot_kws={"size": 14}) # font size


# |TN FN|
# |FP TP|
plt.show()

# COMMAND ----------

y_test_true = testScoreAndLabels_pd["label"]
y_test_pred = testScoreAndLabels_pd["raw"]
conf_mat_test = confusion_matrix(y_test_true, y_test_pred)

print("Test Set:")
print("Accuracy Score: ", accuracy_score(y_test_true, y_test_pred))
print("F1 Score: ", f1_score(y_test_true, y_test_pred))
print(classification_report(y_test_true, y_test_pred))


df_cm_test = pd.DataFrame(conf_mat_test, range(2), range(2))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm_test, annot=True, annot_kws={"size": 14}) # font size


# |TN FN|
# |FP TP|
plt.show()

# COMMAND ----------

model_version = 'rf_model_final'
feat_imp = pd.read_csv(f'/dbfs/mnt/mids-w261/team20SSDK/models/model_meta/{model_version}/feat_imp.csv')
feat_imp.head(30)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### To Be Continued...
# MAGIC 
# MAGIC Some of the next things we hope to try to improve the model is to revisit the outcomes and continue our analysis on the trends we see in incorrect prediction. Looking at things on a per class basis, there are areas for improvement. 
# MAGIC 
# MAGIC Conducting an exhaustive hyper parameter tuning phase, as well as trying new model variants such as **XGBoost** and **Gradient Boosted Trees** are on the roadmap. 
# MAGIC 
# MAGIC Overall, this project has been a huge learning opportunity for our team. From messy raw data, to cleaning, to clustering, to feature engineering, to modelling, and to all of these phases all over again, this project has taken us through a complete data science lifecycle.
# MAGIC 
# MAGIC <em>Until next time, safe travels, and thank you for flying with us! <em>
