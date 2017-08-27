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
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistImagePipelineExampleLoad {

  private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleLoad.class);

  public static void main(String[] args) throws Exception {
    // number of rows and columns in the input pictures
    final int numRows = 28;
    final int numColumns = 28;

    int outputNum = 10; // number of output classes
    int batchSize = 128;  // batch size for each epoch
    int rngSeed = 123;  // random number seed for reproducibility
    int numEpochs = 15; // number of epochs to perform

    // Get the DataSetIterators:
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

    log.info("Load trained model....");

    // where to save model
    File locationToSave = new File("trained_mnist_model.zip");

    MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

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
