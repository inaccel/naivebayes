/**
* Copyright © 2018-2020 InAccel
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

#include "NaiveBayes.h"
#include "util.h"

#define NUMCLASSES_MAX 64 // Max number of model classes
#define NUMFEATURES_MAX 2047 // Max number of model features

#define NUM_REQUESTS 8 // Number of requests for InAccel Coral
#define VECTORIZATION 16 // Vectorization of features in HW
#define PARALLELISM 8 // Parallelism for chunkSize in HW

NaiveBayes::NaiveBayes(int numClasses, int numFeatures, int threads): numClasses(numClasses) {
	assert (numClasses <= NUMCLASSES_MAX);
	assert (numFeatures <= NUMFEATURES_MAX);

	omp_set_num_threads(threads);

	this->numFeatures = (numFeatures + (VECTORIZATION - 1)) & (~(VECTORIZATION - 1));

	priors.resize(numClasses);
	means.resize(numClasses * this->numFeatures);
	variances.resize(numClasses * this->numFeatures);

	_priors.resize(NUM_REQUESTS);
	_means.resize(NUM_REQUESTS);
	_variances.resize(NUM_REQUESTS);

	for (int i = 0; i < NUM_REQUESTS; i++) {
		_priors[i].resize(numClasses);
		_means[i].resize(numClasses * this->numFeatures);
		_variances[i].resize(numClasses * this->numFeatures);
	}

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

	features.resize(NUM_REQUESTS);
	for (int i = 0; i < NUM_REQUESTS; i++) {
		features[i].resize(chunkSize * numFeatures);
	}

	std::ifstream train;
	train.open(filename.c_str());

	std::string line;
	int n = 0, i = 0, c = 0;

	while (getline(train, line) && (n < numExamples)) {
		if (line.length()) {
			if (n && !(n % chunkSize)) {
				c++;
				i = 0;
			}

			std::vector<std::string> tokens = split(line);

			labels[n] = atoi(tokens[0].c_str());

			for (int j = 0; j < numFeatures; j++) {
				features[c][i * numFeatures + j] = atof(tokens[j + 1].c_str());
			}

			n++;
			i++;
		}
	}

	_features.resize(numExamples);

	i = 0; c = 0;
	for (int n = 0; n < numExamples; n++) {
		_features[n].resize(numFeatures);

		c = n / chunkSize;
		i = n % chunkSize;

		for (int j = 0; j < numFeatures; j++) {
			_features[n][j] = features[c][i * numFeatures + j];
		}
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

	int c = 0;
	for (int n = 0, i = 0; n < labels.size(); n++, i++) {
		int label = labels[n];
		class_cnt[label]++;

		if (n && !(n % chunkSize)) {
			c++;
			i = 0;
		}

		for (int j = 0; j < numFeatures; j++) {
			float data = features[c][i * numFeatures + j];
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

	predictions.resize(NUM_REQUESTS);
	for (int i = 0; i < NUM_REQUESTS; i++) {
		predictions[i].resize(chunkSize);
	}

	if (hw) classifyHW(epsilon);
	else classifySW(epsilon);

	auto end = std::chrono::high_resolution_clock::now();

	float seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
	std::cout << "took: " << seconds << "s\n";
}

void NaiveBayes::classifySW(float epsilon) {
	#pragma omp parallel for
	for (int n = 0; n < labels.size(); n++) {
		float max_likelihood = -INFINITY;

		int c = n / chunkSize;
		int i = n % chunkSize;

		for (int k = 0; k < numClasses; k++) {
			float numerator = log(priors[k]);
			for (int j = 0; j < numFeatures; j++) {
				numerator += log(1 / sqrt(2 * M_PI * (variances[k * numFeatures + j] + epsilon))) + ((-1 * (_features[n][j] - means[k * numFeatures + j]) * (_features[n][j] - means[k * numFeatures + j])) / (2 * (variances[k * numFeatures + j] + epsilon)));
			}

			if (numerator > max_likelihood) {
				max_likelihood = numerator;
				predictions[c][i] = k;
			}
		}
	}
}

void NaiveBayes::classifyHW(float epsilon) {
	for (int n = 0; n < NUM_REQUESTS; n++) {
		for (int k = 0; k < numClasses; k++) {
			_priors[n][k] = priors[k];
			for (int j = 0; j < numFeatures; j++) {
				_means[n][k * numFeatures + j] = means[k * numFeatures + j];
				_variances[n][k * numFeatures + j] = variances[k * numFeatures + j];
			}
		}
	}

	std::vector<inaccel::request> nbc;
	std::vector<inaccel::session> sessions(NUM_REQUESTS);
	for (int n = 0; n < NUM_REQUESTS; n++) {
		nbc.push_back(inaccel::request{"com.inaccel.ml.NaiveBayes.Classifier"});

		nbc[n].arg(features[n])
			.arg(_means[n])
			.arg(_variances[n])
			.arg(_priors[n])
			.arg(predictions[n])
			.arg(epsilon)
			.arg(numClasses)
			.arg(numFeatures)
			.arg(chunkSize);
	}

	for (int n = 0; n < NUM_REQUESTS; n++) {
		sessions[n] = inaccel::submit(nbc[n]);
	}

	for (int n = 0; n < NUM_REQUESTS; n++) {
		inaccel::wait(sessions[n]);
	}
}

void NaiveBayes::predict(float epsilon, int hw) {
	classify(epsilon, hw);

	int cor = 0, total = 0;
	int c = 0;
	for (int n = 0, i = 0; n < labels.size(); n++, i++) {
		if (n && !(n % chunkSize)) {
			c++;
			i = 0;
		}

		if (predictions[c][i] == labels[n]) cor++;

		total++;
	}

	std::cout << "\n -- Accuracy: " << (100 * (float)(cor) / total) << " % (" << cor << "/" << total << ")\n\n";
}
