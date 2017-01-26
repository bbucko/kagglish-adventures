package pl.bbucko.spark.kaggle.titanic;

import com.google.common.collect.ImmutableMap;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

public class SparkSolver {

    public static void main(String[] args) throws Exception {
        final SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .master("local")
                .getOrCreate();

        final SparkContext sc = spark.sparkContext();
        final JavaSparkContext context = new JavaSparkContext(sc);

        Dataset<Row> trainData = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/test/resources/train.csv");

        //Calculate median
        double median = trainData.stat().approxQuantile("Age", new double[]{0.5}, 0.25)[0];

        //cleanup trainData
        trainData = trainData.na().fill(median, new String[]{"Age"})
                .na().fill("S", new String[]{"Embarked"})
                .na().replace("Sex", ImmutableMap.of("male", "0", "female", "1"))
                .na().replace("Embarked", ImmutableMap.of("S", "0", "C", "1", "N", "2", "Q", "3"));

        trainData.sparkSession().udf().register("featuresUDT", (UDF4<Double, String, String, Double, Vector>) SparkSolver::toVec, SQLDataTypes.VectorType());
        trainData.sparkSession().udf().register("labelUDT", (UDF1<Integer, Double>) Double::valueOf, DataTypes.DoubleType);

        trainData.show();

        final Dataset<Row> training = trainData
                .withColumn("features", callUDF("featuresUDT", col("Age"), col("Sex"), col("Embarked"), col("Fare")))
                .withColumn("label", callUDF("labelUDT", col("Survived")))
                .select("features", "label");

        final LogisticRegression lr = new LogisticRegression();
        final LogisticRegressionModel model = lr.fit(training);
        System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");

        final Dataset<Row> testData = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/test/resources/test.csv")
                .na().fill(median, new String[]{"Age"})
                .na().fill("S", new String[]{"Embarked"})
                .na().replace("Sex", ImmutableMap.of("male", "0", "female", "1"))
                .na().replace("Embarked", ImmutableMap.of("S", "0", "C", "1", "N", "2", "Q", "3"))
                .withColumn("features", callUDF("featuresUDT", col("Age"), col("Sex"), col("Embarked"), col("Fare")))
                .withColumn("label", col("PassengerId"))
                .select("features", "label");

        Dataset<Row> results = model.transform(testData);

        results.show();

        context.stop();
    }

    private static Vector toVec(Double age, String sex, String embarked, Double fare) {
        return Vectors.dense(age, Double.valueOf(sex), Double.valueOf(embarked), fare);
    }
}

