#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <queue>
#include <vector>

namespace py = pybind11;

extern "C" {
int hyperiblt_decode(int numRow, int numCol, int64_t numel, uint32_t modulo,
                     py::array_t<int64_t> keySum, py::array_t<float> valSum,
                     py::array_t<int32_t> counter,
                     py::array_t<int64_t> hashBuckets,
                     py::array_t<float> results) {
  // std::cout << "Hello from hyperiblt_decode" << std::endl;
  auto keySumProxy = keySum.mutable_unchecked<2>();     // (numRow, numCol)
  auto valSumProxy = valSum.mutable_unchecked<2>();     // (numRow, numCol)
  auto counterProxy = counter.mutable_unchecked<2>();   // (numRow, numCol)
  auto hashProxy = hashBuckets.mutable_unchecked<2>();  // (numRow, numNumel)
  auto resultProxy = results.mutable_unchecked<1>();
  // std::cout << "Ok ... " << std::endl;

  auto isPureBucket = [&](int i, int j, int64_t &key) {
    int32_t cnt = counterProxy(i, j);
    if (keySumProxy(i, j) % cnt != 0) {
      return false;
    }
    key = keySumProxy(i, j) / cnt;

    // NOTE (ozr): may be buggy here ...
    if (key >= numel || key < 0) {
      return false;
    }
    if (key < numel && hashProxy(i, key) != j) {
      return false;
    }
    return true;
  };

  auto removeFromSlot = [&keySumProxy, &valSumProxy, &counterProxy](
                            int r, int c, int pureRow, int pureCol) {
    keySumProxy(r, c) -= keySumProxy(pureRow, pureCol);
    valSumProxy(r, c) -= valSumProxy(pureRow, pureCol);
    counterProxy(r, c) -= counterProxy(pureRow, pureCol);
  };

  // std::cout << "generating candidates: numRow: " << numRow
  //           << "numCol: " << numCol << std::endl;

  std::queue<int> *candidates = new std::queue<int>[numRow];
  std::vector<std::vector<bool>> visited(numRow);
  for (int i = 0; i < numRow; i++) {
    visited[i].resize(numCol);
    for (int j = 0; j < numCol; j++) {
      visited[i][j] = false;
    }
  }

  // std::cout << "generating candidates: Ok ..." << std::endl;

  int64_t key = 0;

  for (int i = 0; i < numRow; i++) {
    for (int j = 0; j < numCol; j++) {
      if (counterProxy(i, j) == 0) {
        // std::cout << "searching for (" << i << ", " << j << ")" << std::endl;
        // std::cout << "visited " << std::endl;
        visited[i][j] = true;
      } else if (isPureBucket(i, j, key)) {
        // std::cout << "searching for (" << i << ", " << j << ")" << std::endl;
        // std::cout << "pure bucket with key = " << key << std::endl;
        visited[i][j] = true;
        resultProxy(key) += valSumProxy(i, j);

        for (int k = 0; k < numRow; k++) {
          if (k == i) {
            continue;
          }

          int64_t pos = hashProxy(k, key);
          removeFromSlot(k, pos, i, j);

          candidates[k].push(pos);
        }

        removeFromSlot(i, j, i, j);
      }
    }
  }

  bool stop = true;
  while (true) {
    stop = true;

    // std::cout << "Hello from Loop..." << std::endl;

    for (int i = 0; i < numRow; i++) {
      while (!candidates[i].empty()) {
        int j = candidates[i].front();
        candidates[i].pop();

        // std::cout << "Row " << i << " : current candidate: " << j
        //           << " numcol: " << numCol << std::endl;

        if (visited[i][j]) {
          // std::cout << "candidate: " << j << " visited " << std::endl;
          continue;
        }

        // std::cout << "checking counterProxy: " << j << std::endl;
        if (counterProxy(i, j) == 0) {
          visited[i][j] = true;
          continue;
        }

        // std::cout << "visiting pure bucket " << j << std::endl;
        // std::cout << "visiting pure bucket " << j
        //           << " value = " << keySumProxy(i, j) / counterProxy(i, j)
        //           << std::endl;

        if (isPureBucket(i, j, key)) {
          visited[i][j] = true;
          // std::cout << "current key" << key << std::endl;

          resultProxy(key) += valSumProxy(i, j);

          for (int k = 0; k < numRow; k++) {
            if (k == i) continue;
            int64_t pos = hashProxy(k, key);

            // std::cout << "Moving " << pos << " to candidates." << std::endl;

            removeFromSlot(k, pos, i, j);
            candidates[k].push(pos);
          }

          removeFromSlot(i, j, i, j);
        }

        // std::cout << "A fail key: " << key << std::endl;
      }
    }

    for (int i = 0; i < numRow; i++) {
      if (!candidates[i].empty()) {
        stop = false;
      }
    }

    if (stop) {
      break;
    }
  }

  int failCnt = 0;
  for (int i = 0; i < numRow; i++) {
    for (int j = 0; j < numCol; j++) {
      if (!visited[i][j]) {
        failCnt++;
      }
    }
  }

  delete[] candidates;

  return failCnt;
}
}

PYBIND11_MODULE(fed_iblt, m) {
  m.doc() = "HyperIBLT by pybind11";

  m.def("hyperiblt_decode", &hyperiblt_decode);
}
