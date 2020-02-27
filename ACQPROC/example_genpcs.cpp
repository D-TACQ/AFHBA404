#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cerr << "USAGE: filname\n";
		return -1;
	}

	std::ifstream i(argv[1]);
	json j;
	i >> j;

//	std::cout << j["AFHBA"]["UUT"];

	int nai = 0;
	for (auto uut : j["AFHBA"]["UUT"]) {
		std::cout << "uut:" << uut["name"] << " type:" << uut["type"] << " AI offset:" << nai << std::endl;
		try {
			nai += uut["VI"]["AI16"].get<int>();
		} catch (std::exception& e){
			std::cerr << "uut:" << uut["name"] << " has no AI16 " << std::endl;
		}
	}

	// write prettified JSON to another file
	std::ofstream o("pretty.json");
	o << std::setw(4) << j << std::endl;
}


