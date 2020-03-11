package com.inaccel.ml;

import com.inaccel.coral.InAccel;
import com.inaccel.coral.msg.Request;
import com.inaccel.coral.shm.SharedIntVector;
import com.inaccel.coral.shm.SharedFloatMatrix;
import com.inaccel.coral.shm.SharedFloatVector;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class NaiveBayes {
	private static final int NUMCLASSES_MAX = 64; // Max number of model classes
	private static final int NUMFEATURES_MAX = 2047; // Max number of model features

	private static final int NUM_REQUESTS = 8; // Number of requests for InAccel Coral
	private static final int VECTORIZATION = 16; // Vectorization of features in HW
	private static final int PARALLELISM = 8; // Parallelism for chunkSize in HW

	private int [] labels;
	private SharedFloatMatrix [] features = new SharedFloatMatrix[NUM_REQUESTS];

	private float [] priors;
	private float [][] means;
	private float [][] variances;

	private SharedFloatVector [] _priors = new SharedFloatVector[NUM_REQUESTS];
	private SharedFloatMatrix [] _means = new SharedFloatMatrix[NUM_REQUESTS];
	private SharedFloatMatrix [] _variances = new SharedFloatMatrix[NUM_REQUESTS];

	private SharedIntVector [] predictions = new SharedIntVector[NUM_REQUESTS];

	private final int numClasses;
	private final int numFeatures;

	private int chunkSize;

	public NaiveBayes(int numClasses, int numFeatures) {
		this.numClasses = numClasses;
		this.numFeatures = numFeatures;

		priors = new float[numClasses];
		means = new float[numClasses][numFeatures];
		variances = new float[numClasses][numFeatures];

		try {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				_priors[i] = new SharedFloatVector(numClasses).alloc();
				_means[i] = new SharedFloatMatrix(numClasses, numFeatures).setRowAttributes(0, VECTORIZATION).alloc();
				_variances[i] = new SharedFloatMatrix(numClasses, numFeatures).setRowAttributes(0, VECTORIZATION).alloc();
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

		int c = 0;
		for (int n = 0, i = 0; n < labels.length; n++, i++) {
			int label = labels[n];
			class_cnt[label]++;

			if ((n != 0) && ((n % chunkSize) == 0)) {
				c++;
				i = 0;
			}

			for (int j = 0; j < numFeatures; j++) {
				float data = features[c].get(i,j);
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

			if (predictions[c].get(i) == labels[n]) cor++;

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

		try {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				features[i] = new SharedFloatMatrix(chunkSize, numFeatures).setRowAttributes(0, 16).setColAttributes(0, 8).alloc();
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
					features[c].put(i, j, Float.parseFloat(tokens[j + 1]));
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
				predictions[i] = new SharedIntVector(chunkSize).alloc();
			}
		} catch(Exception e) {
			System.err.println("Could not allocate shared buffer. Is InAccel Coral running?");
			e.printStackTrace();
			System.exit(-1);
		}

		for (int n = 0; n < NUM_REQUESTS; n++) {
			for (int k = 0; k < numClasses; k++) {
				_priors[n].put(k, priors[k]);
				for (int j = 0; j < numFeatures; j++) {
					_means[n].put(k, j, means[k][j]);
					_variances[n].put(k, j, variances[k][j]);
				}
			}
		}

		Request [] requests = new Request[NUM_REQUESTS];

		for (int n = 0; n < NUM_REQUESTS; n++) {
			requests[n] = new Request("com.inaccel.ml.NaiveBayes.Classifier")
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

		InAccel [] sessions = new InAccel[NUM_REQUESTS];

		try {
			for (int n = 0; n < NUM_REQUESTS; n++) {
				sessions[n] = InAccel.submit(requests[n]);
			}
		} catch(Exception e) {
			System.err.println("Could not submit requests to InAccel Coral");
			e.printStackTrace();
			System.exit(-1);
		}

		try {
			for (int n = 0; n < NUM_REQUESTS; n++) {
				InAccel.wait(sessions[n]);
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
