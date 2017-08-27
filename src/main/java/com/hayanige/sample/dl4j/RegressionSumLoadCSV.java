/**
 * Original work Copyright 2017 Skymind Inc.
 * Modified work Copyright 2017 hayanige
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.hayanige.sample.dl4j;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegressionSumLoadCSV {

  private static Logger log = LoggerFactory.getLogger(RegressionSumLoadCSV.class);

  //  Random number generator seed, for reproducability
  public static final int seed = 12345;
  // Number of iterations per minibatch
  public static final int iterations = 1;
  //  Number of epochs (full passes of the data)
  public static final int nEpochs = 2000;
  //  Number of data points
  public static final int nSamples = 1000;
  //  Batch size: i.e., each epoch has nSamples/batchSize parameter updates
  public static final int batchSize = 100;
  //  Network learning rate
  public static final double learningRate = 0.01;
  // The range of the sample data, data in range (0-1) is sensitive for NN, you
  // can try other ranges and see how it effects the results. Also try changing
  // the range along with changing the activation function
  public static int MIN_RANGE = 0;
  public static int MAX_RANGE = 3;

  public static void main(String[] args) throws Exception {

    // Generate the training data
    Random rng = new Random(seed);
    DataSetIterator iterator = getTrainingData(batchSize, rng);

    // Create the network
    int numInput = 2;
    int numOutputs = 1;
    int nHidden = 10;
    MultiLayerNetwork net = new MultiLayerNetwork(
      new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS) // To configure: .updater(new Nesterovs(0.9))
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
        .activation(Activation.TANH).build())
      .layer(1, new OutputLayer.Builder(LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(nHidden).nOut(numOutputs).build())
      .pretrain(false).backprop(true).build()
    );
    net.init();
    net.setListeners(new ScoreIterationListener(1));

    // Train the network on the full data set, and evaluate in periodically
    for (int i = 0; i < nEpochs; i++) {
      iterator.reset();
      net.fit(iterator);
    }

    // Test the addition of 2 numbers (Try different numbers here)
    final INDArray input = Nd4j.create(new double[] {0.111111, 0.3333333333333},
      new int[] {1, 2});
    INDArray out = net.output(input, false);
    System.out.println(out);
  }

  private static DataSetIterator getTrainingData(int batchSize, Random rand)
    throws Exception {

    log.info("***** load data from csv *****");
    RecordReader recordReader = new CSVRecordReader(0, ",");

    // load training samples
    File trainingFile = new File("src/main/resources/sum_training_samples.csv");
    recordReader.initialize(new FileSplit(trainingFile));
    DataSetIterator trainingIterator = new RecordReaderDataSetIterator(
        recordReader, nSamples);
    DataSet trainings = trainingIterator.next();

    // load training labels
    File labelFile = new File("src/main/resources/sum_training_labels.csv");
    recordReader.initialize(new FileSplit(labelFile));
    DataSetIterator labelIterator = new RecordReaderDataSetIterator(
        recordReader, nSamples);
    DataSet labels = labelIterator.next();

    DataSet dataSet = new DataSet(trainings.getFeatureMatrix(), labels.getFeatureMatrix());
    List<DataSet> listDs = dataSet.asList();
    Collections.shuffle(listDs, rand);
    return new ListDataSetIterator(listDs, batchSize);
  }
}
