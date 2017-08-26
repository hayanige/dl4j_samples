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

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MLPMnistSingleLayerExample {

  private static Logger log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.class);

  public static void main(String[] args) throws Exception {
    // number of rows and columns in the input pictures
    final int numRows = 28;
    final int numColumns = 28;

    int outputNum = 10; // number of output classes
    int batchSize = 128;  // batch size for each epoch
    int rngSeed = 123;  // random number seed for reproducibility
    int numEpochs = 15; // number of epochs to perform

    // Get the DataSetIterators:
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

    log.info("Build model....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)  // include a random seed for reproducibility
      //  use stochastic gradient descent as an optimization algorithm
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.006)  // specify the learning rate
      .updater(Updater.NESTEROVS)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder() // create the first, input layer with xavier initialization
          .nIn(numRows * numColumns).nOut(1000).activation(Activation.RELU)
          .weightInit(WeightInit.XAVIER).build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
          .nIn(1000).nOut(outputNum).activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER).build())
      .pretrain(false).backprop(true) // use backpropagation to adjust weights
      .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    // print the score with every 1 iteration
    model.setListeners(new ScoreIterationListener(1));

    log.info("Train model ....");
    for (int i = 0; i < numEpochs; i++) {
      model.fit(mnistTrain);
    }

    log.info("Evaluate model....");
    Evaluation eval = new Evaluation(outputNum);  // create an evaluation object with 10 possible classes
    while(mnistTest.hasNext()) {
      DataSet next = mnistTest.next();
      INDArray output = model.output(next.getFeatures());  // get the networks prediction
      eval.eval(next.getLabels(), output);  // check the prediction against the true class
    }

    log.info(eval.stats());
    log.info("****Example finished****");
  }

}
