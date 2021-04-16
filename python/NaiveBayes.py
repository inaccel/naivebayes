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
	VECTORIZATION = 16;
	PARALLELISM = 8;

	def __init__(self, numClasses, numFeatures):
		self.numClasses = numClasses
		self.numFeatures = numFeatures

		if (numFeatures % self.VECTORIZATION):
			self.numFeaturesPadded = numFeatures + (self.VECTORIZATION - numFeatures % self.VECTORIZATION)
		else:
			self.numFeaturesPadded = numFeatures

		self.priors = np.ndarray(numClasses, dtype = np.float32)
		self.means = np.ndarray((numClasses, numFeatures), dtype = np.float32)
		self.variances = np.ndarray((numClasses, numFeatures), dtype = np.float32)

		self._priors = []
		self._means = []
		self._variances = []

		with inaccel.allocator:
			for x in range(0, self.NUM_REQUESTS):
				self._priors.append(np.ndarray(numClasses, dtype = np.float32))
				self._means.append(np.ndarray(numClasses * self.numFeaturesPadded, dtype = np.float32))
				self._variances.append(np.ndarray(numClasses * self.numFeaturesPadded, dtype = np.float32))

	def load_data(self, filename, numExamples):
		print("\n -- Reading input file ", end='')

		self.labels = np.ndarray(numExamples, dtype = np.int32)

		self.chunkSize = int(numExamples / self.NUM_REQUESTS)
		if (numExamples % self.NUM_REQUESTS):
			self.chunkSize += 1

		if (self.chunkSize % self.PARALLELISM):
			self.chunkSize = int(self.chunkSize + (self.PARALLELISM - self.chunkSize % self.PARALLELISM))

		self.features = []

		with inaccel.allocator:
			for x in range(0, self.NUM_REQUESTS):
				self.features.append(np.ndarray(self.chunkSize * self.numFeaturesPadded, dtype = np.float32))

		self._features = np.ndarray((numExamples, self.numFeaturesPadded), dtype = np.float32)

		start = int(round(time.time() * 100))

		with open(filename) as fp:
			lines = fp.readlines()

			c = 0
			i = 0
			n = 0
			for line in lines:
				tokens = line.split(',')

				if (n == numExamples):
					break
				if (i == self.chunkSize):
					c += 1
					i = 0

				self.labels[n] = tokens[0]

				tokens = tokens[1:]
				self._features[n][:] = tokens[:]

				idx = i * self.numFeaturesPadded
				self.features[c][idx:idx+self.numFeatures] = self._features[n][:]

				n += 1
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

		c = 0
		i = 0
		for n in range(0, self.labels.size):
			label = self.labels[n]
			class_cnt[label] += 1

			sums[label][:] += self._features[n][:]
			sq_sums[label][:] += self._features[n][:] * self._features[n][:]

		for k in range(0, self.numClasses):
			self.priors[k] = float(class_cnt[k]) / float(self.numFeatures)

			cls_cnt = float(class_cnt[k])
			for j in range(0, self.numFeatures):
				self.means[k][j] = sums[k][j] / cls_cnt
				sq_feature_means[k][j] = sq_sums[k][j] / cls_cnt
				self.variances[k][j] = sq_feature_means[k][j] - (self.means[k][j] * self.means[k][j])

		end = int(round(time.time() * 100))

		print("took: " + str((end - start) / 100) + "s")

	def classify(self, epsilon):
		print("\n -- Classification ", end='')

		start = int(round(time.time() * 100))

		self.predictions = []

		with inaccel.allocator:
			for i in range(0, self.NUM_REQUESTS):
				self.predictions.append(np.ndarray(self.chunkSize, dtype = np.int32))

		for n in range(0, self.NUM_REQUESTS):
			self._priors[n][:] = self.priors[:]
			for k in range(0, self.numClasses):
				idx = k * self.numFeaturesPadded

				self._means[n][idx:idx + self.numFeatures] = self.means[k][:];
				self._variances[n][idx:idx + self.numFeatures] = self.variances[k][:];

		responses = []
		for n in range(0, self.NUM_REQUESTS):
			req = inaccel.request("com.inaccel.ml.NaiveBayes.Classifier")

			req.arg(self.features[n]).arg(self._means[n]).arg(self._variances[n]).arg(self._priors[n]).arg(self.predictions[n]).arg(np.float32(epsilon)).arg(np.int32(self.numClasses)).arg(np.int32(self.numFeatures)).arg(np.int32(self.chunkSize))

			responses.append(inaccel.submit(req))

		for n in range(0, self.NUM_REQUESTS):
			responses[n].result()

		end = int(round(time.time() * 100))

		print("took: " + str((end - start) / 100) + "s")

	def predict(self, epsilon):
		self.classify(epsilon)

		cor = 0
		total = 0
		c = 0
		i = 0

		for n in range(0, self.labels.size):
			if ((n != 0) and ((n % self.chunkSize) == 0)):
				c += 1
				i = 0

			if (self.predictions[c][i] == self.labels[n]):
				cor += 1

			total += 1
			i += 1

		print("\n -- Accuracy:  " + str(round(100 * float(cor) / total, 2)) + " % (" + str(cor) + "/" + str(total) + ")\n")
