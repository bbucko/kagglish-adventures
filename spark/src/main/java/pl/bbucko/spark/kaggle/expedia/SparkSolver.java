package pl.bbucko.spark.kaggle.expedia;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

public class SparkSolver {
    public static void main(String[] args) {
        final SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .master("local")
                .getOrCreate();

        final SparkContext sc = spark.sparkContext();
        final JavaSparkContext context = new JavaSparkContext(sc);
    }
}
