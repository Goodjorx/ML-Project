from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import RobustScaler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import SQLContext

spark = SparkSession.builder.appName("MLapp").getOrCreate()
# Loading Dataset
df = spark.read.csv("/home/goodie/Desktop/POLIBA/Big Data/ML Project/data/predictive_maintenance.csv", header=True, inferSchema=True)
df.cache()

# Starting Exploratory Data Analysis, EDA

df.printSchema()
#df.show(5)
# Filter Null values
# print(col, "\t", "with null values: ", df.filter(df[col].isNull()).count())
# print(col, "\t", "with ? values: ", df.filter(df[col] == '?').count())
# To use seaborn, drop UID and Product ID.
df_p = df.toPandas()
df_p.drop(df.columns[:2], axis=1, inplace=True)

# Visualize Data. Histograms and BoxPlots. Look for outliers or skewed data.
'''for i, col in enumerate(df_p.columns, 1):
    #plt.subplot(3,3,i)
    if col == 'Type' or col == 'Failure Type' or col == 'Target':
        sns.countplot(x = str(col), data = df_p)
        plt.savefig('Images/'+col+'_BoxPlot.png')
        plt.cla()
    else:
        sns.boxplot(y=col, x = "Target", data=df_p)
        plt.savefig('Images/'+col + '_BoxPlot.png')
        plt.cla()
        sns.histplot(df_p[col], kde=True)
        plt.savefig('Images/'+col+'_Histogram.png')
        plt.cla()

# sns.heatmap(df_p.corr())
# plt.savefig("Images/HeatMap.png")
# plt.cla()
'''

# Splitting Dataset 70% - 30%
(train, test) = df.randomSplit([0.7, 0.3], seed=42)

# ML Pipeline
"""skewed = QuantileDiscretizer(numBuckets=10, inputCol='Air temperature [K]', outputCol='Air temperature [K]_binned')
df_binned = skewed.fit(df).transform(df)
#df_binned.show(10)
x = df_binned.select('Air temperature [K]_binned').toPandas()
plt.hist(x)
plt.savefig('Images/AirTemperature_Skewed.png')
plt.cla()"""
# Index Type col. We also need to index the Failure in order to use a Multiclassifier.
indexers = StringIndexer(inputCols=['Type', 'Failure Type'], outputCols=['Type_indexed', 'Failure_idx'])
# Hot encode Type col
encoder = OneHotEncoder(inputCols=['Type_indexed'], outputCols=['Type_encoded'])
# Assemble all features
assembler_features = VectorAssembler(
    inputCols=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
               'Torque [Nm]', 'Tool wear [min]', 'Type_encoded'],
    outputCol='features')

# Binary Classifier. Predict if Failure
svc = LinearSVC(featuresCol='features', labelCol='Target')
pipeline = Pipeline(stages=[indexers, encoder, assembler_features, svc])

# Build GRID for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(svc.maxIter, [100, 250, 500])  # model iteration
             .addGrid(svc.aggregationDepth, [2, 3])
             .build())

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol='Target',
                          metricName='f1'),
                          numFolds=5,
                          parallelism=2)
cvModel = crossval.fit(train)
predSVC = cvModel.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol='Target')
print("For binary Classifier: Weighted Precision = %g" % evaluator.evaluate(predSVC, {evaluator.metricName: "weightedPrecision"}))
print("For binary Classifier: Weighted Recall = %g" % evaluator.evaluate(predSVC, {evaluator.metricName: "weightedRecall"}))
print("For binary Classifier: F1 = %g" % evaluator.evaluate(predSVC, {evaluator.metricName: "f1"}))
print('For binary Classifier: Acc:', evaluator.evaluate(predSVC, {evaluator.metricName: "accuracy"}))
print('For binary Classifier: TPR:', evaluator.evaluate(predSVC, {evaluator.metricName: "truePositiveRateByLabel"}))
print('For binary Classifier: FPR:', evaluator.evaluate(predSVC, {evaluator.metricName: "falsePositiveRateByLabel"}))
print('For binary Classifier: Precision:', evaluator.evaluate(predSVC, {evaluator.metricName: "precisionByLabel"}))
print("-------------------------------\n\n-------------------------------")
# MultiLabel Classifier.
NB = NaiveBayes(featuresCol='features', labelCol='Failure_idx')
pipeline = Pipeline(stages=[indexers, encoder, assembler_features, NB])

# Build GRID for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(NB.modelType, ['multinomial', 'gaussian']) #model type
             .addGrid(NB.smoothing, [1, 0.8, 0.65])
             .build())

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol='Failure_idx',                      metricName='f1'),
                          numFolds=5,
                          parallelism=2)
cvModel = crossval.fit(train) # selects best model depending on evaluator
predNB = cvModel.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol='Failure_idx')

print("For MultiLabel Classifier: Weighted Precision = %g" % evaluator.evaluate(predNB, {evaluator.metricName: "weightedPrecision"}))
print("For MultiLabel Classifier: Weighted Recall = %g" % evaluator.evaluate(predNB, {evaluator.metricName: "weightedRecall"}))
print("For MultiLabel Classifier: F1 = %g" % evaluator.evaluate(predNB, {evaluator.metricName: "f1"}))
print('For MultiLabel Classifier: Acc:', evaluator.evaluate(predNB, {evaluator.metricName: "accuracy"}))
print('For MultiLabel Classifier: TPR:', evaluator.evaluate(predNB, {evaluator.metricName: "truePositiveRateByLabel"}))
print('For MultiLabel Classifier: FPR:', evaluator.evaluate(predNB, {evaluator.metricName: "falsePositiveRateByLabel"}))
print('For MultiLabel Classifier: Precision:', evaluator.evaluate(predNB, {evaluator.metricName: "precisionByLabel"}))


spark.stop()
