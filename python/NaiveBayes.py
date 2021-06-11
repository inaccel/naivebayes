# Copyright © 2018-2021 InAccel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inaccel.coral as inaccel
import numpy as np
import time

class NB:
	NUMCLASSES_MAX = 64;
	NUMFEATURES_MAX = 2047;

	NUM_REQUESTS = 8;
	VECTORIZATION = 8;
	PARALLELISM = 4096;

	def __init__(self, numClasses, numFeatures):
		self.numClasses = numClasses
		self.numFeatures = numFeatures

		if (numFeatures % self.VECTORIZATION):
			self.numFeaturesPadded = numFeatures + (self.VECTORIZATION - numFeatures % self.VECTORIZATION)
		else:
			self.numFeaturesPadded = numFeatures

		with inaccel.allocator:
			self.priors = np.ndarray(numClasses, dtype = np.float32)
			self.means = np.ndarray((numClasses, self.numFeaturesPadded), dtype = np.float32)
			self.variances = np.ndarray((numClasses, self.numFeaturesPadded), dtype = np.float32)

	def load_data(self, filename, numExamples):
		print("\n -- Reading input file ", end='')

		self.labels = np.ndarray(numExamples, dtype = np.int32)

		self.chunkSize = int(numExamples / self.NUM_REQUESTS)
		if (numExamples % self.NUM_REQUESTS):
			self.chunkSize += 1

		if (self.chunkSize % self.PARALLELISM):
			self.chunkSize = int(self.chunkSize + (self.PARALLELISM - self.chunkSize % self.PARALLELISM))

		with inaccel.allocator:
			self.features = np.ndarray((self.NUM_REQUESTS * self.chunkSize, self.numFeaturesPadded), dtype = np.float32)

		start = int(round(time.time() * 100))

		with open(filename) as fp:
			lines = fp.readlines()

			i = 0
			for line in lines:
				tokens = line.split(',')

				if (i == numExamples):
					break

				self.labels[i] = tokens[0]
				self.features[i][:self.numFeatures] = tokens[1:]

				i += 1

		end = int(round(time.time() * 100))

		print("took: " + str((end - start) / 100) + "s");

	def train(self, filename, numExamples):
		self.load_data(filename, numExamples)

		print("\n -- Training ", end='')

		start = int(round(time.time() * 100))

		class_cnt = np.zeros(self.numClasses, dtype = np.int32)

		sums = np.zeros((self.numClasses, self.numFeatures), dtype = np.float32)
		sq_sums = np.zeros((self.numClasses, self.numFeatures), dtype = np.float32)
		sq_feature_means = np.ndarray((self.numClasses, self.numFeatures), dtype = np.float32)

		for i in range(0, self.labels.size):
			label = self.labels[i]
			class_cnt[label] += 1

			sums[label][:] += self.features[i][:self.numFeatures]
			sq_sums[label][:] += self.features[i][:self.numFeatures] * self.features[i][:self.numFeatures]

		self.priors[:] = class_cnt[:] / float(self.numFeatures)
		for k in range(0, self.numClasses):
			self.means[k] = sums[k] / float(class_cnt[k])
			self.variances[k][:] = (sq_sums[k] / float(class_cnt[k])) - self.means[k][:] * self.means[k][:]

		end = int(round(time.time() * 100))

		print("took: " + str((end - start) / 100) + "s")

	def classify(self, epsilon):
		print("\n -- Classification ", end='')

		start = int(round(time.time() * 100))

		with inaccel.allocator:
			self.predictions = np.ndarray((self.NUM_REQUESTS, self.chunkSize), dtype = np.int32)

		responses = []
		for n in range(0, self.NUM_REQUESTS):
			req = inaccel.request("com.inaccel.ml.NaiveBayes.Classifier")

			req.arg(self.features.reshape(self.NUM_REQUESTS, self.chunkSize * self.numFeaturesPadded)[n][:]) \
				.arg(self.means) \
				.arg(self.variances) \
				.arg(self.priors) \
				.arg(self.predictions[n][:]) \
				.arg(np.float32(epsilon)) \
				.arg(np.int32(self.numClasses)) \
				.arg(np.int32(self.numFeatures)) \
				.arg(np.int32(self.chunkSize))

			responses.append(inaccel.submit(req))

		for n in range(0, self.NUM_REQUESTS):
			responses[n].result()

		end = int(round(time.time() * 100))

		print("took: " + str((end - start) / 100) + "s")

	def predict(self, epsilon):
		self.classify(epsilon)

		cor = 0

		for i in range(0, self.labels.size):
			if (self.predictions.reshape(self.NUM_REQUESTS * self.chunkSize)[i] == self.labels[i]):
				cor += 1

		print("\n -- Accuracy:  " + str(round(100 * float(cor) / self.labels.size, 2)) + " % (" + str(cor) + "/" + str(self.labels.size) + ")\n")
