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

#include <assert.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>

#include "NaiveBayes.h"

#define NUMCLASSES_MAX 64 // Max number of model classes
#define NUMFEATURES_MAX 2047 // Max number of model features

#define NUM_REQUESTS 8 // Number of requests for InAccel Coral
#define VECTORIZATION 8 // Vectorization of features in HW
#define PARALLELISM 4096 // Parallelism for chunkSize in HW

NaiveBayes::NaiveBayes(int numClasses, int numFeatures, int threads): numClasses(numClasses) {
	assert (numClasses <= NUMCLASSES_MAX);
	assert (numFeatures <= NUMFEATURES_MAX);

	omp_set_num_threads(threads);

	this->numFeatures = numFeatures;
	this->numFeaturesPadded = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

	priors.resize(numClasses);
	means.resize(numClasses * this->numFeaturesPadded);
	variances.resize(numClasses * this->numFeaturesPadded);

	std::cout << std::fixed;
	std::cout << std::setprecision(2);
}

void NaiveBayes::load_data(std::string filename, int numExamples) {
	std::cout << "\n -- Reading Input File " << std::flush;

	auto start = std::chrono::high_resolution_clock::now();
	labels.resize(numExamples);

	chunkSize = numExamples / NUM_REQUESTS;
	if (numExamples % NUM_REQUESTS) chunkSize++;

	chunkSize = (chunkSize + (PARALLELISM - 1)) & (~(PARALLELISM - 1));

	features.resize(NUM_REQUESTS * chunkSize * numFeaturesPadded);

	std::ifstream train;
	train.open(filename.c_str());

	std::string line;
	std::string token;
	int i = 0;

	while (getline(train, line) && (i < numExamples)) {
		std::stringstream linestream(line);

		getline(linestream, token, ',');
		labels[i] = std::stoi(token);

		for (int j = 0; j < numFeatures; j++) {
			getline(linestream, token, ',');
			features[i * numFeaturesPadded + j] = std::stof(token);
		}

		i++;
	}

	train.close();

	auto end = std::chrono::high_resolution_clock::now();

	float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
	std::cout << "took: " << seconds << "s\n";
}

void NaiveBayes::train(std::string filename, int numExamples) {
	load_data(filename, numExamples);

	std::cout << "\n -- Training " << std::flush;

	auto start = std::chrono::high_resolution_clock::now();

	int *class_cnt = (int *) malloc(numClasses * sizeof(int));

	float *sums, *sq_sums, *sq_feature_means;
	sums = (float *) malloc(numClasses * numFeatures * sizeof(float));
	sq_sums = (float *) malloc(numClasses * numFeatures * sizeof(float));
	sq_feature_means = (float *) malloc(numClasses * numFeatures * sizeof(float));

	for (int k = 0; k < numClasses; k++) {
		class_cnt[k] = 0;

		for (int j = 0; j < numFeatures; j++) {
			sums[k * numFeatures + j] = 0;
			sq_sums[k * numFeatures + j] = 0;
		}
	}

	for (int i = 0; i < labels.size(); i++) {
		int label = labels[i];
		class_cnt[label]++;

		for (int j = 0; j < numFeatures; j++) {
			float data = features[i * numFeaturesPadded + j];
			sums[label * numFeatures + j] += data;
			sq_sums[label * numFeatures + j] += data * data;
		}
	}

	for (int k = 0; k < numClasses; k++) {
		priors[k] = class_cnt[k] / (float)numFeatures;

		for (int j = 0; j < numFeatures; j++) {
			means[k * numFeatures + j] = sums[k * numFeatures + j] / (float)class_cnt[k];
			sq_feature_means[k * numFeatures + j] = sq_sums[k * numFeatures + j] / (float)class_cnt[k];
			variances[k * numFeatures + j] = sq_feature_means[k * numFeatures + j] - (means[k * numFeatures + j] * means[k * numFeatures + j]);
		}
	}

	free(sums);
	free(sq_sums);
	free(sq_feature_means);

	auto end = std::chrono::high_resolution_clock::now();

	float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
	std::cout << "took: " << seconds << "s\n";
}

void NaiveBayes::classify(float epsilon, int hw) {
	std::cout << "\n -- Classification " << std::flush;

	auto start = std::chrono::high_resolution_clock::now();

	predictions.resize(NUM_REQUESTS * chunkSize);

	if (hw) classifyHW(epsilon);
	else classifySW(epsilon);

	auto end = std::chrono::high_resolution_clock::now();

	float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
	std::cout << "took: " << seconds << "s\n";
}

void NaiveBayes::classifySW(float epsilon) {
	#pragma omp parallel for
	for (int i = 0; i < labels.size(); i++) {
		float max_likelihood = -INFINITY;

		for (int k = 0; k < numClasses; k++) {
			float numerator = log(priors[k]);
			for (int j = 0; j < numFeatures; j++) {
				numerator += log(1 / sqrt(2 * M_PI * (variances[k * numFeatures + j] + epsilon))) + ((-1 * (features[i * numFeatures + j] - means[k * numFeatures + j]) * (features[i * numFeatures + j] - means[k * numFeatures + j])) / (2 * (variances[k * numFeatures + j] + epsilon)));
			}

			if (numerator > max_likelihood) {
				max_likelihood = numerator;
				predictions[i] = k;
			}
		}
	}
}

void NaiveBayes::classifyHW(float epsilon) {
	std::vector<std::future<void>> responses(NUM_REQUESTS);
	for (int n = 0; n < NUM_REQUESTS; n++) {
		inaccel::request nbc("com.inaccel.ml.NaiveBayes.Classifier");

		nbc.arg<float>(features.begin() + n * chunkSize * numFeatures, features.begin() + (n + 1) * chunkSize * numFeatures)
			.arg(means)
			.arg(variances)
			.arg(priors)
			.arg<int>(predictions.begin() + n * chunkSize, predictions.begin() + (n + 1) * chunkSize)
			.arg(epsilon)
			.arg(numClasses)
			.arg(numFeatures)
			.arg(chunkSize);

		responses[n] = inaccel::submit(nbc);
	}

	for (int n = 0; n < NUM_REQUESTS; n++) {
		responses[n].get();
	}
}

void NaiveBayes::predict(float epsilon, int hw) {
	classify(epsilon, hw);

	int cor = 0;
	for (int i = 0; i < labels.size(); i++) {
		if (predictions[i] == labels[i]) cor++;
	}

	std::cout << "\n -- Accuracy: " << (100 * (float)(cor) / labels.size()) << " % (" << cor << "/" << labels.size() << ")\n\n";
}
