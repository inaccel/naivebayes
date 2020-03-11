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
