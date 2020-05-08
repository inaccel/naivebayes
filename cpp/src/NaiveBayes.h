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

#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include <inaccel/coral>
#include <string>

class NaiveBayes {
private:
	int numClasses;
	int numFeatures;
	int chunkSize;

	std::vector<int> labels;
	std::vector<inaccel::vector<float>> features;
	std::vector<std::vector<float>> _features;

	inaccel::vector<float> priors;
	inaccel::vector<float> means;
	inaccel::vector<float> variances;

	std::vector<inaccel::vector<float>> _priors;
	std::vector<inaccel::vector<float>> _means;
	std::vector<inaccel::vector<float>> _variances;

	std::vector<inaccel::vector<int>> predictions;

	void load_data(std::string filename, int numExamples);

	void classify(float epsilon, int hw);

	void classifySW(float epsilon);

	void classifyHW(float epsilon);

public:
	NaiveBayes(int numClasses, int numFeatures, int threads);

	void train(std::string filename, int numExamples);

	void predict(float epsilon, int hw);
};

#endif // NAIVEBAYES_H
