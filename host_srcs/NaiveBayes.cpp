/*
Copyright © 2019 InAccel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _TEST_
#define _accel_ 1
#else
#define _accel_ 0
#endif

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <sys/time.h>
#include <vector>

#include <coral-api/coral.h>

#define NUMCLASSES 26      // Number of classes
#define NUMFEATURES 784    // number of feaures
#define NUMEXAMPLES 124800 // Number of training/test data
#define NUM_REQUESTS 8     // Number of requests for InAccel Coral
#define CHUNKSIZE (NUMEXAMPLES / NUM_REQUESTS)

using namespace std;

// Function to split a string on specified delimiter
vector<string> split(const string &s) {
  vector<string> elements;
  stringstream ss(s);
  string item;

  while (getline(ss, item)) {
    size_t prev = 0;
    size_t pos;

    while ((pos = item.find_first_of(" (,[])=", prev)) != std::string::npos) {
      if (pos > prev)
        elements.push_back(item.substr(prev, pos - prev));
      prev = pos + 1;
    }

    if (prev < item.length())
      elements.push_back(item.substr(prev, std::string::npos));
  }

  return elements;
}

// Reads the input dataset and sets features and labels buffers accordingly
void read_input(string filename, std::vector<int> &labels,
                std::vector<inaccel::vector<float>> &features) {
  ifstream train;
  train.open(filename.c_str());

  string line;
  int n = 0, i = 0, c = 0;

  while (getline(train, line) && (n < NUMEXAMPLES)) {
    if (line.length()) {
      if (n && !(n % CHUNKSIZE)) {
        c++;
        i = 0;
      }
      vector<string> tokens = split(line);

      labels[n] = atoi(tokens[0].c_str());
      for (int j = 0; j < NUMFEATURES; j++) {
        features[c][i * NUMFEATURES + j] = atof(tokens[j + 1].c_str());
      }

      n++;
      i++;
    }
  }

  train.close();
}

void NBtraining(std::vector<int> &labels,
                std::vector<inaccel::vector<float>> &features,
                inaccel::vector<float> &priors, inaccel::vector<float> &means,
                inaccel::vector<float> &variances) {
  int *class_cnt = (int *)malloc(NUMCLASSES * sizeof(int));
  float *sums, *sq_sums, *sq_feature_means;
  sums = (float *)malloc(NUMCLASSES * NUMFEATURES * sizeof(float));
  sq_sums = (float *)malloc(NUMCLASSES * NUMFEATURES * sizeof(float));
  sq_feature_means = (float *)malloc(NUMCLASSES * NUMFEATURES * sizeof(float));

  for (int k = 0; k < NUMCLASSES; k++) {
    class_cnt[k] = 0;
    for (int j = 0; j < NUMFEATURES; j++) {
      sums[k * NUMFEATURES + j] = 0;
      sq_sums[k * NUMFEATURES + j] = 0;
    }
  }

  int c = 0;
  for (int n = 0, i = 0; n < NUMEXAMPLES; n++, i++) {

    int label = labels[n];
    class_cnt[label]++;

    if (n && !(n % CHUNKSIZE)) {
      c++;
      i = 0;
    }

    for (int j = 0; j < NUMFEATURES; j++) {
      float data = features[c][i * NUMFEATURES + j];
      sums[label * NUMFEATURES + j] += data;
      sq_sums[label * NUMFEATURES + j] += data * data;
    }
  }

  for (int k = 0; k < NUMCLASSES; k++) {
    priors[k] = class_cnt[k] / (float)NUMEXAMPLES;
    for (int j = 0; j < NUMFEATURES; j++) {
      means[k * NUMFEATURES + j] =
          sums[k * NUMFEATURES + j] / (float)class_cnt[k];
      sq_feature_means[k * NUMFEATURES + j] =
          sq_sums[k * NUMFEATURES + j] / (float)class_cnt[k];
      variances[k * NUMFEATURES + j] =
          sq_feature_means[k * NUMFEATURES + j] -
          (means[k * NUMFEATURES + j] * means[k * NUMFEATURES + j]);
    }
  }

  free(sums);
  free(sq_sums);
  free(sq_feature_means);
}

void NBclassify(std::vector<inaccel::vector<float>> &features,
                inaccel::vector<float> &means,
                inaccel::vector<float> &variances,
                inaccel::vector<float> &priors, float epsilon,
                std::vector<inaccel::vector<int>> &predictions) {
  int c = 0;
  float d_Pi = 2 * M_PI;

  for (int n = 0, i = 0; n < NUMEXAMPLES; n++, i++) {
    float max_likelihood = -INFINITY;
    if (n && !(n % CHUNKSIZE)) {
      c++;
      i = 0;
    }

    for (int k = 0; k < NUMCLASSES; k++) {
      float numerator = log(priors[k]);
      for (int j = 0; j < NUMFEATURES; j++) {
        numerator +=
            log(1 / sqrt(d_Pi * (variances[k * NUMFEATURES + j] + epsilon))) +
            ((-1 *
              (features[c][i * NUMFEATURES + j] - means[k * NUMFEATURES + j]) *
              (features[c][i * NUMFEATURES + j] - means[k * NUMFEATURES + j])) /
             (2 * (variances[k * NUMFEATURES + j] + epsilon)));
      }

      if (numerator > max_likelihood) {
        max_likelihood = numerator;
        predictions[c][i] = k;
      }
    }
  }
}

int main(int argc, const char *argv[]) {
  struct timeval startRead, endRead, startTrain, endTrain, startTest, endTest;

  std::vector<int> labels(NUMEXAMPLES);

  std::vector<inaccel::vector<float>> features(NUM_REQUESTS);
  for (int i = 0; i < NUM_REQUESTS; i++) {
    features[i].resize(CHUNKSIZE * NUMFEATURES);
  }

  inaccel::vector<float> priors(NUMCLASSES);
  inaccel::vector<float> means(NUMCLASSES * NUMFEATURES);
  inaccel::vector<float> variances(NUMCLASSES * NUMFEATURES);

  std::vector<inaccel::vector<int>> predictions(NUM_REQUESTS);
  for (int i = 0; i < NUM_REQUESTS; i++) {
    predictions[i].resize(CHUNKSIZE);
  }

  std::vector<inaccel::vector<float>> _priors(NUM_REQUESTS);
  for (int i = 0; i < NUM_REQUESTS; i++) {
    _priors[i].resize(NUMCLASSES);
  }

  std::vector<inaccel::vector<float>> _means(NUM_REQUESTS);
  for (int i = 0; i < NUM_REQUESTS; i++) {
    _means[i].resize(NUMCLASSES * NUMFEATURES);
  }

  std::vector<inaccel::vector<float>> _variances(NUM_REQUESTS);
  for (int i = 0; i < NUM_REQUESTS; i++) {
    _variances[i].resize(NUMCLASSES * NUMFEATURES);
  }

  if (argc != 1) {
    cout << "Usage: ./" << argv[0] << endl;
    exit(-1);
  }

  // Specify train and test input files as well as output model file
  string inFile = "data/letters_csv_train.dat";

  gettimeofday(&startRead, NULL);
  read_input(inFile, labels, features);
  gettimeofday(&endRead, NULL);

  float time_us = ((endRead.tv_sec * 1000000) + endRead.tv_usec) -
                  ((startRead.tv_sec * 1000000) + startRead.tv_usec);
  float time_s = (endRead.tv_sec - startRead.tv_sec);
  cout << "! Time reading input data: " << time_us / 1000 << " msec, " << time_s
       << " sec " << endl;

  // Train a NaiveBayes model
  cout << "    * NaiveBayes Training *" << endl;

  gettimeofday(&startTrain, NULL);

  // Invoke the software implementation of the algorithm
  NBtraining(labels, features, priors, means, variances);

  gettimeofday(&endTrain, NULL);

  time_us = ((endTrain.tv_sec * 1000000) + endTrain.tv_usec) -
            ((startTrain.tv_sec * 1000000) + startTrain.tv_usec);
  time_s = (endTrain.tv_sec - startTrain.tv_sec);
  cout << "        ! Time for training model: " << time_us / 1000 << " msec, "
       << time_s << " sec " << endl;

  // Compute the accuracy of the trained model on the same train dataset.
  cout << "    * NaiveBayes Testing *" << endl;

  for (int n = 0; n < NUM_REQUESTS; n++) {
    for (int k = 0; k < NUMCLASSES; k++) {
      _priors[n][k] = priors[k];
      for (int j = 0; j < NUMFEATURES; j++) {
        _means[n][k * NUMFEATURES + j] = means[k * NUMFEATURES + j];
        _variances[n][k * NUMFEATURES + j] = variances[k * NUMFEATURES + j];
      }
    }
  }

  int cor = 0, total = 0;

  float epsilon = 0.05;

  if (_accel_) {
    gettimeofday(&startTest, NULL);

    std::vector<inaccel::Request> nbc;
    for (int n = 0; n < NUM_REQUESTS; n++) {
      nbc.push_back(inaccel::Request{"com.inaccel.ml.NaiveBayes.Classifier"});
      nbc[n]
          .Arg(features[n])
          .Arg(_means[n])
          .Arg(_variances[n])
          .Arg(_priors[n])
          .Arg(predictions[n])
          .Arg(epsilon)
          .Arg(NUMCLASSES)
          .Arg(NUMFEATURES)
          .Arg(CHUNKSIZE);
    }

    for (int n = 0; n < NUM_REQUESTS; n++) {
      inaccel::Coral::SubmitAsync(nbc[n]);
    }

    for (int n = 0; n < NUM_REQUESTS; n++) {
      inaccel::Coral::Await(nbc[n]);
    }

  } else {
    gettimeofday(&startTest, NULL);

    NBclassify(features, means, variances, priors, epsilon, predictions);
  }

  int c = 0;
  for (int n = 0, i = 0; n < NUMEXAMPLES; n++, i++) {
    if (n && !(n % CHUNKSIZE)) {
      c++;
      i = 0;
    }

    if (predictions[c][i] == labels[n])
      cor++;

    total++;
  }

  gettimeofday(&endTest, NULL);

  time_us = ((endTest.tv_sec * 1000000) + endTest.tv_usec) -
            ((startTest.tv_sec * 1000000) + startTest.tv_usec);
  time_s = (endTest.tv_sec - startTest.tv_sec);
  cout << "        ! Time for evaluating model accuracy: " << time_us / 1000
       << " msec, " << time_s << " sec " << endl;

  printf("        - Accuracy: %3.2f %% (%i/%i)\n", (100 * (float)(cor) / total),
         cor, total);

  return EXIT_SUCCESS;
}
