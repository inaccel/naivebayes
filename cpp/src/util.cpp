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
