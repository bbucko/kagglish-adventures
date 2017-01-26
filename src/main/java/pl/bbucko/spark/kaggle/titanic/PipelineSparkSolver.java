package pl.bbucko.spark.kaggle.titanic;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class PipelineSparkSolver {

    public static void main(String[] args) throws Exception {
        final SparkSession spark = SparkSession
                .builder()
                .appName("TitanicPipelineSolver")
                .master("local")
                .getOrCreate();

        final SparkContext sc = spark.sparkContext();
        final JavaSparkContext context = new JavaSparkContext(sc);

        Dataset<Row> training = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/test/resources/train.csv")
                .withColumnRenamed("Survived", "label");

        double median = training.stat().approxQuantile("Age", new double[]{0.5}, 0.25)[0];

        training = training
                .na().fill("S", new String[]{"Embarked"})
                .na().fill(median, new String[]{"Age"});

        training.printSchema();
        training.show();

        StringIndexer sexIndexer = new StringIndexer()
                .setInputCol("Sex")
                .setOutputCol("SexIndex");

        StringIndexer embarkedIndexer = new StringIndexer()
                .setInputCol("Embarked")
                .setOutputCol("EmbarkedIndex");

        VectorAssembler featuresAssembler = new VectorAssembler()
                .setInputCols(new String[]{"SexIndex", "Age", "bucketedFare", "EmbarkedIndex"})
                .setOutputCol("features");

        double[] splits = {Double.NEGATIVE_INFINITY, 0, 15, 100, Double.POSITIVE_INFINITY};

        Bucketizer fareBucketizer = new Bucketizer()
                .setInputCol("Fare")
                .setOutputCol("bucketedFare")
                .setSplits(splits);


        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.001);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{sexIndexer, embarkedIndexer, fareBucketizer, featuresAssembler, lr});

        PipelineModel model = pipeline.fit(training);

        Dataset<Row> test = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/test/resources/test.csv")
                .na().fill("S", new String[]{"Embarked"})
                .na().fill(median, new String[]{"Age"});

        Dataset<Row> predictions = model.transform(test);
        predictions.show();

        context.stop();
    }
}

