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

#include <iostream>
#include <string>

#include "NaiveBayes.h"

int main(int argc, const char *argv[]) {
	if (argc != 3) {
		std::cout << "Usage: ./" << argv[0] << " <CPU threads> <HW/SW, SW:0, HW:1>\n";
		exit(-1);
	}

	const uint threads = std::atoi(argv[1]);
	const uint hw = std::atoi(argv[2]);

	NaiveBayes nb(26, 784, threads);

	nb.train(std::string(std::getenv("HOME")) + "/data/letters_csv_train.dat", 124800);

	float epsilon = 0.05;

	nb.predict(epsilon, hw);

	return 0;
}
