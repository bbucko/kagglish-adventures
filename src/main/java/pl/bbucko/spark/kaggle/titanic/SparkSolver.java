package pl.bbucko.spark.kaggle.titanic;

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
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.types.DataTypes;

import java.util.HashMap;
import java.util.Map;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

public class SparkSolver {

    final private static Map<String, String> sexMap = new HashMap<>();

    static {
        sexMap.put("male", "0");
        sexMap.put("female", "1");
    }


    final private static Map<String, String> embarkedMap = new HashMap<>();

    static {
        embarkedMap.put("S", "0");
        embarkedMap.put("C", "1");
        embarkedMap.put("N", "2");
        embarkedMap.put("Q", "3");
    }

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
                .na().replace("Sex", sexMap)
                .na().replace("Embarked", embarkedMap);

        trainData.sparkSession().udf().register("featuresUDT", (UDF3<Double, String, String, Vector>) SparkSolver::toVec, SQLDataTypes.VectorType());
        trainData.sparkSession().udf().register("labelUDT", (UDF1<Integer, Double>) Double::valueOf, DataTypes.DoubleType);

        trainData.show();

        final Dataset<Row> training = trainData
                .withColumn("features", callUDF("featuresUDT", col("Age"), col("Sex"), col("Embarked")))
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
                .na().replace("Sex", sexMap)
                .na().replace("Embarked", embarkedMap)
                .withColumn("features", callUDF("featuresUDT", col("Age"), col("Sex"), col("Embarked")))
                .withColumn("label", col("PassengerId"))
                .select("features", "label");

        Dataset<Row> results = model.transform(testData);

        results.show();

        context.stop();
    }

    private static Vector toVec(Double age, String sex, String embarked) {
        return Vectors.dense(age, Double.valueOf(sex), Double.valueOf(embarked));
    }
}

