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

#include <sstream>

#include "util.h"

// Function to split a string on specified delimiter
std::vector<std::string> split(const std::string &s) {
	std::vector<std::string> elements;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item)) {
		size_t prev = 0;
		size_t pos;

		while ((pos = item.find_first_of(" (,[])=", prev)) != std::string::npos) {
			if (pos > prev) elements.push_back(item.substr(prev, pos - prev));
			prev = pos + 1;
		}

		if (prev < item.length()) elements.push_back(item.substr(prev, std::string::npos));
	}

	return elements;
}
