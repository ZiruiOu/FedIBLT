
bind:
	g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` bind.cpp -o HyperIBLT_cpp`python3-config --extension-suffix` -I ./fedIBLT/
