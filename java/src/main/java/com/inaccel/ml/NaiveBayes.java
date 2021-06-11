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
	private static final int PARALLELISM = 4096; // Parallelism for chunkSize in HW

	private int [] labels;
	private ByteBuf features;
	private ByteBuf priors;
	private ByteBuf means;
	private ByteBuf variances;
	private ByteBuf predictions;

	private final int numClasses;
	private final int numFeatures;
	private final int numFeaturesPadded;

	private int chunkSize;

	public NaiveBayes(int numClasses, int numFeatures) {
		this.numClasses = numClasses;
		this.numFeatures = numFeatures;
		numFeaturesPadded = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

		try {
			priors = InAccelByteBufAllocator.DEFAULT.buffer(numClasses * Float.BYTES);
			means = InAccelByteBufAllocator.DEFAULT.buffer(numClasses * numFeaturesPadded * Float.BYTES);
			variances = InAccelByteBufAllocator.DEFAULT.buffer(numClasses * numFeaturesPadded * Float.BYTES);
		} catch (Exception e) {
			System.err.println("Could not allocate shared buffers. Is InAccel service running?");
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

		for (int k = 0; k < numClasses; k++) {
			class_cnt[k] = 0;

			for (int j = 0; j < numFeatures; j++) {
				sums[k][j] = 0;
				sq_sums[k][j] = 0;
			}
		}

		for (int i = 0; i < labels.length; i++) {
			int label = labels[i];
			class_cnt[label]++;

			for (int j = 0; j < numFeatures; j++) {
				float data = features.getFloat(Float.BYTES * (i * numFeaturesPadded + j));
				sums[label][j] += data;
				sq_sums[label][j] += data * data;
			}
		}

		for (int k = 0; k < numClasses; k++) {
			priors.setFloat(Float.BYTES * k, class_cnt[k] / (float) numFeatures);

			for (int j = 0; j < numFeatures; j++) {
				means.setFloat(Float.BYTES * (k * numFeaturesPadded + j), sums[k][j] / (float)class_cnt[k]);
				variances.setFloat(Float.BYTES * (k * numFeaturesPadded + j), (sq_sums[k][j] / (float)class_cnt[k]) - (means.getFloat(Float.BYTES * (k * numFeaturesPadded + j)) * means.getFloat(Float.BYTES * (k * numFeaturesPadded + j))));
			}
		}

		long end = System.nanoTime();

		double duration = (end - start) / (double) 1000000000;
		System.out.println(String.format("took: %.2fs", duration));
	}

	public void predict(float epsilon) {
		classify(epsilon);

		int cor = 0;
		for (int i = 0; i < labels.length; i++) {
			if (predictions.getInt(Integer.BYTES * i) == labels[i]) cor++;
		}

		System.out.println(String.format("\n -- Accuracy: %.2f %% (%d/%d)\n", (100 * (float)(cor) / labels.length), cor, labels.length));
	}

	private void load_data(String filename, int numExamples) {
		System.out.print("\n -- Reading Input File ");

		long start = System.nanoTime();

		labels = new int[numExamples];

		chunkSize = numExamples / NUM_REQUESTS;
		if ((numExamples % NUM_REQUESTS) != 0) chunkSize++;

		chunkSize = (chunkSize + (PARALLELISM - 1)) & (~(PARALLELISM - 1));

		try {
			features = InAccelByteBufAllocator.DEFAULT.buffer(NUM_REQUESTS * chunkSize * numFeaturesPadded * Float.BYTES);
		} catch (Exception e) {
			System.err.println("Could not allocate shared buffers. Is InAccel service running?");
			e.printStackTrace();
			System.exit(-1);
		}

		try {
			BufferedReader reader = new BufferedReader(new FileReader(filename));

			String line;
			int i = 0;

			while (((line = reader.readLine()) != null) && (i < numExamples)) {
				String [] tokens = line.split(",");

				labels[i] = Integer.parseInt(tokens[0]);
				for (int j = 0; j < numFeatures; j++) {
					features.setFloat(Float.BYTES * (i * numFeaturesPadded + j), Float.parseFloat(tokens[j + 1]));
				}

				i++;
			}

			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		long end = System.nanoTime();

		double duration = (end - start) / (double) 1000000000;
		System.out.println(String.format("took: %.2fs", duration));
	}

	private void classify(float epsilon){
		System.out.print("\n -- Classification ");

		long start = System.nanoTime();

		try {
			predictions = InAccelByteBufAllocator.DEFAULT.buffer(NUM_REQUESTS * chunkSize * Integer.BYTES);
		} catch(Exception e) {
			System.err.println("Could not allocate shared buffer. Is InAccel service running?");
			e.printStackTrace();
			System.exit(-1);
		}

		List<Future<Void>> responses = new ArrayList<>(NUM_REQUESTS);
		try {
			for (int n = 0; n < NUM_REQUESTS; n++) {
				InAccel.Request request = new InAccel.Request("com.inaccel.ml.NaiveBayes.Classifier")
				.arg(features.slice(Float.BYTES * n * chunkSize * numFeaturesPadded, Float.BYTES * chunkSize * numFeaturesPadded))
				.arg(means)
				.arg(variances)
				.arg(priors)
				.arg(predictions.slice(Integer.BYTES * n * chunkSize, Float.BYTES * chunkSize))
				.arg(epsilon)
				.arg(numClasses)
				.arg(numFeatures)
				.arg(chunkSize);

				responses.add(n, InAccel.submit(request));
			}
		} catch(Exception e) {
			System.err.println("Could not submit requests to InAccel Coral");
			e.printStackTrace();
			System.exit(-1);
		}

		try {
			for (Future<Void> response: responses) {
				response.get();
			}
		} catch(Exception e) {
			System.err.println("Error waiting on requests to finish!");
			e.printStackTrace();
			System.exit(-1);
		}

		long end = System.nanoTime();

		double duration = (end - start) / (double) 1000000000;
		System.out.println(String.format("took: %.2fs", duration));
	}
}
