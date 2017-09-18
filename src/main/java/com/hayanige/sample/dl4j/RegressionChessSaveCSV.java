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
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegressionChessSaveCSV {

  private static Logger log = LoggerFactory.getLogger(RegressionChessSaveCSV.class);

  //  Random number generator seed, for reproducability
  public static final int seed = 12345;
  // Number of iterations per minibatch
  public static int iterations = 1;
  //  Number of epochs (full passes of the data)
  public static int nEpochs = 1000;
  //  Number of data points
  public static int nSamples = 1000;
  //  Batch size: i.e., each epoch has nSamples/batchSize parameter updates
  public static int batchSize = 100;
  //  Network learning rate
  public static double learningRate = 0.000001;

  public static void main(String[] args) throws Exception {

    Options opts = createOption();
    CommandLine cli = new DefaultParser().parse(opts, args);

    if (cli.hasOption("h")) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("score generator from fen", opts);
      return;
    }

    if (cli.hasOption("i")) {
      iterations = Integer.parseInt(cli.getOptionValue("i"));
    }

    if (cli.hasOption("e")) {
      nEpochs = Integer.parseInt(cli.getOptionValue("e"));
    }

    if (cli.hasOption("s")) {
      nSamples = Integer.parseInt(cli.getOptionValue("s"));
    }

    if (cli.hasOption("b")) {
      batchSize = Integer.parseInt(cli.getOptionValue("b"));
    }

    if (cli.hasOption("l")) {
      learningRate = Double.parseDouble(cli.getOptionValue("l"));
    }

    // Generate the training data
    Random rng = new Random(seed);
    DataSetIterator iterator = getTrainingData(batchSize, rng);

    // Create the network
    int numInput = 65;
    int numOutputs = 1;
    int nHidden = 10000;
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

    // Initialize the user interface backend
    UIServer uiServer = UIServer.getInstance();

    // Configure where the network information (gradients, score vs. time etc)
    // is to be stored. Here: store in memory.
    // Alternative: new File StatsStorage(File), for saving and loading later
    StatsStorage statsStorage = new InMemoryStatsStorage();

    // Attatch the StatsStorage instance to the UI: this allows the contents of
    // the StatsStorage to be visualized
    uiServer.attach(statsStorage);

    // Then add the StatsListener to collect this information from the network,
    // as it trains
    net.setListeners(new StatsListener(statsStorage));

    // where to save model
    File locationToSave = new File("trained_chess_model.zip");

    // boolean save Updater
    boolean saveUpdater = false;

    // Train the network on the full data set, and evaluate in periodically
    for (int i = 1; i <= nEpochs; i++) {
      iterator.reset();
      net.fit(iterator);
      if (nEpochs % 10 == 0) {
        log.info("Save trained model....");
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);
      }
    }

    log.info("Save trained model....");
    ModelSerializer.writeModel(net, locationToSave, saveUpdater);
    log.info("=== END ===");
  }

  private static DataSetIterator getTrainingData(int batchSize, Random rand)
    throws Exception {

    log.info("***** load data from csv *****");
    RecordReader recordReader = new CSVRecordReader(0, ",");

    // load training samples
    File trainingFile = new File("training_samples.csv");
    recordReader.initialize(new FileSplit(trainingFile));
    DataSetIterator trainingIterator = new RecordReaderDataSetIterator(
        recordReader, nSamples);
    DataSet trainings = trainingIterator.next();

    // load training labels
    File labelFile = new File("training_labels.csv");
    recordReader.initialize(new FileSplit(labelFile));
    DataSetIterator labelIterator = new RecordReaderDataSetIterator(
        recordReader, nSamples);
    DataSet labels = labelIterator.next();

    DataSet dataSet = new DataSet(trainings.getFeatureMatrix(), labels.getFeatureMatrix());
    List<DataSet> listDs = dataSet.asList();
    Collections.shuffle(listDs, rand);
    return new ListDataSetIterator(listDs, batchSize);
  }

  private static Options createOption() {
    Options result = new Options();

    result.addOption(Option.builder("i")
        .longOpt("iteration")
        .desc("number of iterations")
        .hasArg()
        .build()
    );

    result.addOption(Option.builder("e")
        .longOpt("epoch")
        .desc("number of epochs")
        .hasArg()
        .build()
    );

    result.addOption(Option.builder("s")
        .longOpt("sample")
        .desc("number of samples")
        .hasArg()
        .build()
    );

    result.addOption(Option.builder("b")
        .longOpt("batch")
        .desc("number of batches")
        .hasArg()
        .build()
    );

    result.addOption(Option.builder("l")
        .longOpt("learning")
        .desc("learning rate")
        .hasArg()
        .build()
    );

    result.addOption(Option.builder("h")
        .longOpt("help")
        .desc("help")
        .hasArg(false)
        .build()
    );

    return result;
  }
}
