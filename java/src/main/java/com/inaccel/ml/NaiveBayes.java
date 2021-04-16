/**
* Copyright © 2018-2021 InAccel
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package com.inaccel.ml;

import com.inaccel.coral.*;

import io.netty.buffer.ByteBuf;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Future;
import java.util.List;

public class NaiveBayes {
	private static final int NUMCLASSES_MAX = 64; // Max number of model classes
	private static final int NUMFEATURES_MAX = 2047; // Max number of model features

	private static final int NUM_REQUESTS = 8; // Number of requests for InAccel Coral
	private static final int VECTORIZATION = 8; // Vectorization of features in HW
	private static final int PARALLELISM = 8; // Parallelism for chunkSize in HW

	private int [] labels;
	private ByteBuf [] features = new ByteBuf[NUM_REQUESTS];

	private float [] priors;
	private float [][] means;
	private float [][] variances;

	private ByteBuf [] _priors = new ByteBuf[NUM_REQUESTS];
	private ByteBuf [] _means = new ByteBuf[NUM_REQUESTS];
	private ByteBuf [] _variances = new ByteBuf[NUM_REQUESTS];

	private ByteBuf [] predictions = new ByteBuf[NUM_REQUESTS];

	private final int numClasses;
	private final int numFeatures;

	private int chunkSize;

	public NaiveBayes(int numClasses, int numFeatures) {
		this.numClasses = numClasses;
		this.numFeatures = numFeatures;

		priors = new float[numClasses];
		means = new float[numClasses][numFeatures];
		variances = new float[numClasses][numFeatures];

		int numFeaturesPadded = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

		try {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				_priors[i] = InAccelByteBufAllocator.DEFAULT.buffer(numClasses * Float.BYTES);
				_means[i] = InAccelByteBufAllocator.DEFAULT.buffer(numClasses * numFeaturesPadded * Float.BYTES);
				_variances[i] = InAccelByteBufAllocator.DEFAULT.buffer(numClasses * numFeaturesPadded * Float.BYTES);
			}
		} catch (Exception e) {
			System.err.println("Could not allocate shared buffers. Is InAccel Coral running?");
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public void train(String filename, int numExamples) {
		load_data(filename, numExamples);

		System.out.print("\n -- Training ");

		long start = System.nanoTime();

		int [] class_cnt = new int[numClasses];
		float [][] sums = new float [numClasses][numFeatures];
		float [][] sq_sums = new float [numClasses][numFeatures];
		float [][] sq_feature_means = new float [numClasses][numFeatures];

		for (int k = 0; k < numClasses; k++) {
			class_cnt[k] = 0;

			for (int j = 0; j < numFeatures; j++) {
				sums[k][j] = 0;
				sq_sums[k][j] = 0;
			}
		}

		int numFeaturesPadded = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

		int c = 0;
		for (int n = 0, i = 0; n < labels.length; n++, i++) {
			int label = labels[n];
			class_cnt[label]++;

			if ((n != 0) && ((n % chunkSize) == 0)) {
				c++;
				i = 0;
			}

			for (int j = 0; j < numFeatures; j++) {
				float data = features[c].getFloat(Float.BYTES * (i * numFeaturesPadded + j));
				sums[label][j] += data;
				sq_sums[label][j] += data * data;
			}
		}

		for (int k = 0; k < numClasses; k++) {
			priors[k] = class_cnt[k] / (float) numFeatures;

			for (int j = 0; j < numFeatures; j++) {
				means[k][j] = sums[k][j] / (float)class_cnt[k];
				sq_feature_means[k][j] = sq_sums[k][j] / (float)class_cnt[k];
				variances[k][j] = sq_feature_means[k][j] - (means[k][j] * means[k][j]);
			}
		}

		long end = System.nanoTime();

		double duration = (end - start) / (double) 1000000000;
		System.out.println("took: " + String.format("%.2f", duration) + "s");
	}

	public void predict(float epsilon) {
		classify(epsilon);

		int cor = 0, total = 0;
		int c = 0;
		for (int n = 0, i = 0; n < labels.length; n++, i++) {
			if ((n != 0) && ((n % chunkSize) == 0)) {
				c++;
				i = 0;
			}

			if (predictions[c].getInt(Integer.BYTES * i) == labels[n]) cor++;

			total++;
		}

		System.out.println("\n -- Accuracy: " + String.format("%.2f", (100 * (float)(cor) / total)) + "% (" + cor + "/" + total + ")\n");
	}

	private void load_data(String filename, int numExamples) {
		System.out.print("\n -- Reading Input File ");

		long start = System.nanoTime();

		labels = new int[numExamples];

		chunkSize = numExamples / NUM_REQUESTS;
		if ((numExamples % NUM_REQUESTS) != 0) chunkSize++;

		int chunkSizePadded = (chunkSize + (PARALLELISM - 1)) & (~(PARALLELISM - 1));
		int numFeaturesPadded = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

		try {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				features[i] = InAccelByteBufAllocator.DEFAULT.buffer(chunkSizePadded * numFeaturesPadded * Float.BYTES);
			}
		} catch (Exception e) {
			System.err.println("Could not allocate shared buffers. Is InAccel Coral running?");
			e.printStackTrace();
			System.exit(-1);
		}

		try {
			BufferedReader reader = new BufferedReader(new FileReader(filename));

			String line;
			int n = 0, i = 0, c = 0;

			while (((line = reader.readLine()) != null) && (n < numExamples)) {
				if ((n != 0) && ((n % chunkSize) == 0)) {
					c++;
					i = 0;
				}

				String [] tokens = line.split(",");

				labels[n] = Integer.parseInt(tokens[0]);
				for (int j = 0; j < numFeatures; j++) {
					features[c].setFloat(Float.BYTES * (i * numFeaturesPadded + j), Float.parseFloat(tokens[j + 1]));
				}

				n++;
				i++;
			}

			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		long end = System.nanoTime();

		double duration = (end - start) / (double) 1000000000;
		System.out.println("took: " + String.format("%.2f", duration) + "s");
	}

	private void classify(float epsilon){
		System.out.print("\n -- Classification ");

		long start = System.nanoTime();

		try {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				predictions[i] = InAccelByteBufAllocator.DEFAULT.buffer(chunkSize * Integer.BYTES);
			}
		} catch(Exception e) {
			System.err.println("Could not allocate shared buffer. Is InAccel Coral running?");
			e.printStackTrace();
			System.exit(-1);
		}

		int numFeaturesPadded = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

		for (int n = 0; n < NUM_REQUESTS; n++) {
			for (int k = 0; k < numClasses; k++) {
				_priors[n].setFloat(Float.BYTES * k, priors[k]);
				for (int j = 0; j < numFeatures; j++) {
					_means[n].setFloat(Float.BYTES * (k * numFeaturesPadded + j), means[k][j]);
					_variances[n].setFloat(Float.BYTES * (k * numFeaturesPadded + j), variances[k][j]);
				}
			}
		}

		InAccel.Request[] requests = new InAccel.Request[NUM_REQUESTS];

		for (int n = 0; n < NUM_REQUESTS; n++) {
			requests[n] = new InAccel.Request("com.inaccel.ml.NaiveBayes.Classifier")
				.arg(features[n])
				.arg(_means[n])
				.arg(_variances[n])
				.arg(_priors[n])
				.arg(predictions[n])
				.arg(epsilon)
				.arg(numClasses)
				.arg(numFeatures)
				.arg(chunkSize);
		}

	 	List<Future<Void>> responses = new ArrayList<>(NUM_REQUESTS);

		try {
			for (int n = 0; n < NUM_REQUESTS; n++) {
				responses.add(n, InAccel.submit(requests[n]));
			}
		} catch(Exception e) {
			System.err.println("Could not submit requests to InAccel Coral");
			e.printStackTrace();
			System.exit(-1);
		}

		try {
			for (int n = 0; n < NUM_REQUESTS; n++) {
				responses.get(n).get();
			}
		} catch(Exception e) {
			System.err.println("Error waiting on request to finish!");
			e.printStackTrace();
			System.exit(-1);
		}


		long end = System.nanoTime();

		double duration = (end - start) / (double) 1000000000;
		System.out.println("took: " + String.format("%.2f", duration) + "s");
	}
}
