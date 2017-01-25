package com.sabre.spark.helloworld;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;

public class WordCount {


    public static void main(String[] args) throws Exception {
        final SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .master("local")
                .getOrCreate();

        final SparkContext sc = spark.sparkContext();
        final JavaSparkContext context = new JavaSparkContext(sc);

        Dataset<Row> passengerCSV = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/test/resources/test.csv");

        passengerCSV.printSchema();

        double median = passengerCSV.stat().approxQuantile("Age", new double[]{0.5}, 0.25)[0];
        passengerCSV = passengerCSV.na().fill(median, new String[]{"Age"});

        final JavaRDD<Passenger> passengerRDD = passengerCSV
                .javaRDD()
                .map(line -> {
                    final int passengerId = line.getInt(0);
                    final int pclass = line.getInt(1);
                    final String name = line.getString(2);
                    final boolean sex = line.getString(3).equals("male");
                    final double age = line.getDouble(4);
                    final int sibSp = line.getInt(5);
                    final int parch = line.getInt(6);
                    final String ticket = line.getString(7);
                    final double fare = line.getDouble(8);
                    final String cabin = line.getString(9);
                    final int embarked = embarked(line.getString(10));
                    return new Passenger(passengerId, pclass, name, sex, age, sibSp, parch, ticket, fare, cabin, embarked);
                });

        Dataset<Row> peopleDF = spark.createDataFrame(passengerRDD, Passenger.class);
//        peopleDF.createOrReplaceTempView("people");
        peopleDF.printSchema();
        peopleDF.show();

        context.stop();
    }

    private static int embarked(String string) {
        switch (string) {
            case "C":
                return 0;
            case "S":
                return 1;
            case "Q":
                return 2;
            case "N":
                return 3;
        }
        return 0;
    }


    @Value
    @AllArgsConstructor
    public static class Passenger implements Serializable {
        private final int passengerId;
        private final int pclass;
        private final String name;
        private final boolean sex;
        private final double age;
        private final int sibSp;
        private final int parch;
        private final String ticket;
        private final double fare;
        private final String cabin;
        private final int embarked;

    }

}

