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

#include <ap_int.h>
#include <math.h>

#define numClassesMax 64
#define numFeaturesMax 512
#define vectorSize 8
#define chunk 8

typedef ap_int<256> float8;

union {
  int asInt;
  float asFloat;
} converter0, converter1, converter2;

extern "C" {
void Classifier_3(float8 *_features, float8 *_means, float8 *_variances,
                  float *_priors, int *_prediction, float epsilon,
                  int numClasses, int numFeatures, int chunkSize) {
#pragma HLS INTERFACE m_axi port = _features offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = _means offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = _variances offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = _priors offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = _prediction offset = slave bundle = gmem4
#pragma HLS INTERFACE s_axilite port = _features bundle = control
#pragma HLS INTERFACE s_axilite port = _means bundle = control
#pragma HLS INTERFACE s_axilite port = _variances bundle = control
#pragma HLS INTERFACE s_axilite port = _priors bundle = control
#pragma HLS INTERFACE s_axilite port = _prediction bundle = control
#pragma HLS INTERFACE s_axilite port = epsilon bundle = control
#pragma HLS INTERFACE s_axilite port = numClasses bundle = control
#pragma HLS INTERFACE s_axilite port = numFeatures bundle = control
#pragma HLS INTERFACE s_axilite port = chunksize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  int prediction[chunk];
  float d_Pi = 2 * M_PI;
  float priors[numClassesMax], max_likelihood[chunk],
      numerator[numClassesMax][chunk * vectorSize];
  float8 means[numClassesMax][numFeaturesMax],
      variances[numClassesMax][numFeaturesMax], features[chunk][numFeaturesMax];

// Using URAMs for features, means and variances buffers
#pragma HLS resource variable = features core = XPM_MEMORY uram
#pragma HLS resource variable = means core = XPM_MEMORY uram
#pragma HLS resource variable = variances core = XPM_MEMORY uram

// Partitioning the local arrays
#pragma HLS array_partition variable = features complete dim = 1
#pragma HLS array_partition variable = numerator complete dim = 2

  int numFeatures8 =
      (((numFeatures) + (vectorSize - 1)) & (~(vectorSize - 1))) >> 3;
  int numClassesMin = (13 > numClasses) ? 13 : numClasses;

  for (int k = 0; k < numClasses; k++) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline II = 1
    priors[k] = _priors[k];
  }

  for (int k = 0; k < numClasses; k++) {
#pragma HLS loop_tripcount min = 10 max = 10
    for (int j = 0; j < numFeatures8; j++) {
#pragma HLS loop_tripcount min = 98 max = 98
#pragma HLS pipeline II = 1
      means[k][j] = _means[k * numFeatures8 + j];
      variances[k][j] = _variances[k * numFeatures8 + j];
    }
  }

  for (int i = 0; i < chunkSize / chunk; i++) {
#pragma HLS loop_tripcount min = 1250 max = 1250
    int offset = (i * chunk) * numFeatures8;

    for (int c = 0; c < chunk; c++) {
#pragma HLS unroll
      max_likelihood[c] = -INFINITY;
    }

    for (int cj = 0, c = 0, j = 0; cj < chunk * numFeatures8; cj++, j++) {
#pragma HLS loop_tripcount min = 784 max = 784
#pragma HLS pipeline II = 1
      if (j == numFeatures8) {
        j = 0;
        c++;
      }
      features[c][j] = _features[offset + cj];
    }

    for (int k = 0; k < numClasses; k++) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline II = 1
      for (int c = 0; c < chunk; c++) {
        numerator[k][c * vectorSize] = logf(priors[k]);

        for (int t = 1; t < vectorSize; t++) {
          numerator[k][c * vectorSize + t] = 0.0f;
        }
      }
    }

    for (int j = 0; j < numFeatures8; j++) {
#pragma HLS loop_tripcount min = 98 max = 98
      for (int k = 0; k < numClassesMin; k++) {
#pragma HLS loop_tripcount min = 13 max = 13
#pragma HLS pipeline II = 1
        for (int c = 0; c < chunk; c++) {
          for (int t = 0; t < vectorSize; t++) {
            converter0.asInt = features[c][j].range((t + 1) * 32 - 1, t * 32);
            converter1.asInt = means[k][j].range((t + 1) * 32 - 1, t * 32);
            converter2.asInt = variances[k][j].range((t + 1) * 32 - 1, t * 32);

            float dPiVariances = d_Pi * (converter2.asFloat + epsilon);
            float firstGroup = dPiVariances ? 0.5f * logf(dPiVariances) : 0;

            float difSquared =
                (float)(converter0.asFloat - converter1.asFloat) *
                (converter0.asFloat - converter1.asFloat);
            float variancesD = 2.0f * (converter2.asFloat + epsilon);
            float secondGroup = variancesD ? difSquared / variancesD : 0;

            numerator[k][c * vectorSize + t] -= firstGroup + secondGroup;
          }
        }
      }
    }

    for (int k = 0; k < numClasses; k++) {
#pragma HLS loop_tripcount min = 10 max = 10
      for (int c = 0; c < chunk; c++) {
#pragma HLS loop_tripcount min = 8 max = 8
#pragma HLS pipeline II = 1
        float adder0_0 =
            numerator[k][c * vectorSize] + numerator[k][c * vectorSize + 1];
        float adder0_1 =
            numerator[k][c * vectorSize + 2] + numerator[k][c * vectorSize + 3];
        float adder0_2 =
            numerator[k][c * vectorSize + 4] + numerator[k][c * vectorSize + 5];
        float adder0_3 =
            numerator[k][c * vectorSize + 6] + numerator[k][c * vectorSize + 7];

        float adder1_0 = adder0_0 + adder0_1;
        float adder1_1 = adder0_2 + adder0_3;

        float result = adder1_0 + adder1_1;

        if (result > max_likelihood[c]) {
          max_likelihood[c] = result;
          prediction[c] = k;
        }
      }
    }

    for (int c = 0; c < chunk; c++) {
#pragma HLS loop_tripcount min = 8 max = 8
#pragma HLS pipeline II = 1
      _prediction[i * chunk + c] = prediction[c];
    }
  }
}
}
