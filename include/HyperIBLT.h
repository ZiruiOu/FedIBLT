#ifndef _HYPERIBLT_H_
#define _HYPERIBLT_H_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_map>

#include "BOBHash32.h"
#include "mod.h"
#include "prime.h"

namespace py = pybind11;

#define DEBUG_F 0

const int cc = 300000;
const int dd = 6568640;
const int rr = 3;

// use a 16-bit prime, so 2 * a mod PRIME will not overflow
static const uint32_t PRIME_ID = MAXPRIME[24];
static const uint32_t PRIME_FING = MAXPRIME[24];
typedef double vt;

class HyperIBLT {
 public:
  HyperIBLT(int numRow, int numCol, py::array_t<uint32_t> keySum,
            py::array_t<float> valSum, py::array_t<uint32_t> counter,
            py::array_t<uint32_t> hashBuckets) {
    numRow = numRow;
    numCol = numCol;
  }

 private:
  int numRow;
  int numCol;

  py::array_t<uint32_t> keySum;
  py::array_t<float> valSum;
  py::array_t<uint32_t> counter;
  py::array_t<uint32_t> hashBuckets;
};

#endif
