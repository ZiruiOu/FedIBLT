#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <queue>
#include <vector>

namespace py = pybind11;

class HyperIBLT {
 public:
  HyperIBLT(int numRow, int numCol, int numel, uint32_t modulo,
            py::array_t<uint32_t> keySum, py::array_t<float> valSum,
            py::array_t<uint32_t> counter, py::array_t<uint32_t> hashBuckets)
      : numRow(numRow),
        numCol(numCol),
        numel(numel),
        modulo(modulo),
        keySum(keySum),
        valSum(valSum),
        counter(counter),
        hashBuckets(hashBuckets) {}

  ~HyperIBLT() = default;

  bool Decode(py::array_t<float> results) {
    auto keySumProxy = keySum.mutable_unchecked<2>();     // (numRow, numCol)
    auto valSumProxy = valSum.mutable_unchecked<2>();     // (numRow, numCol)
    auto counterProxy = counter.mutable_unchecked<2>();   // (numRow, numCol)
    auto hashProxy = hashBuckets.mutable_unchecked<2>();  // (numRow, numNumel)
    auto resultProxy = results.mutable_unchecked<1>();

    auto isPureBucket = [&](int i, int j, uint32_t &key) {
      uint32_t cnt = counterProxy(i, j);
      if (keySumProxy(i, j) % cnt != 0) {
        return false;
      }
      key = keySumProxy(i, j) / cnt;
      if (key < (uint32_t)numel && hashProxy(i, key) != j) {
        return false;
      }
      return true;
    };

    auto removeFromSlot = [&keySumProxy, &valSumProxy, &counterProxy](
                              int r, int c, int pureRow, int pureCol) {
      keySumProxy(r, c) = keySumProxy(r, c) - keySumProxy(pureRow, pureCol);
      valSumProxy(r, c) -= valSumProxy(pureRow, pureCol);
      counterProxy(r, c) -= counterProxy(pureRow, pureCol);
    };

    std::queue<int> *candidates = new std::queue<int>[numRow];
    std::vector<std::vector<bool>> visited(numRow);
    for (int i = 0; i < numRow; i++) {
      visited[i].resize(numCol);
      for (int j = 0; j < numCol; j++) {
        visited[i][j] = false;
      }
    }

    uint32_t key = 0;

    for (int i = 0; i < numRow; i++) {
      for (int j = 0; j < numCol; j++) {
        if (counterProxy(i, j) == 0) {
          visited[i][j] = true;
        } else if (isPureBucket(i, j, key)) {
          visited[i][j] = true;
          resultProxy(key) += valSumProxy(i, j);

          for (int k = 0; k < numRow; k++) {
            if (k == i) {
              continue;
            }
            uint32_t pos = hashProxy(k, key);
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

      for (int i = 0; i < numRow; i++) {
        if (!candidates[i].empty()) {
          stop = false;
        }

        while (!candidates[i].empty()) {
          int j = candidates[i].front();
          candidates[i].pop();

          if (visited[i][j]) {
            continue;
          }

          if (counterProxy(i, j) == 0) {
            visited[i][j] = true;
          } else if (isPureBucket(i, j, key)) {
            visited[i][j] = true;
            // TODO : insertion

            resultProxy(key) += valSumProxy(i, j);

            for (int k = 0; k < numRow; k++) {
              if (k == i) continue;
              uint32_t pos = hashProxy(k, key);

              removeFromSlot(k, pos, i, j);
              candidates[k].push(pos);
            }

            removeFromSlot(i, j, i, j);
          }
        }
      }

      if (stop) {
        break;
      }
    }

    delete[] candidates;

    int failCnt = 0;
    for (int i = 0; i < numRow; i++) {
      for (int j = 0; j < numCol; j++) {
        if (!visited[i][j]) {
          failCnt++;
        }
      }
    }

    std::cout << "number of fail counts: " << failCnt << std::endl;
    return (failCnt == 0);
  }

 private:
  int numRow;
  int numCol;
  int numel;
  uint32_t modulo;
  py::array_t<uint32_t> keySum;
  py::array_t<float> valSum;
  py::array_t<uint32_t> counter;
  py::array_t<uint32_t> hashBuckets;
};

PYBIND11_MODULE(fedIBLT, m) {
  m.doc() = "HyperIBLT by pybind11";

  //  pybind11::class_<Fermat>(m, "Fermat")
  //      .def(pybind11::init())
  //      .def("insert", &Fermat::insert)
  //      .def("insert_one", &Fermat::insert_one)
  //      .def("delete_in_one_bucket", &Fermat::delete_in_one_bucket)
  //      .def("verify", &Fermat::verify)
  //      .def("display", &Fermat::display)
  //      .def("decode", &Fermat::decode);

  pybind11::class_<HyperIBLT>(m, "HyperIBLT")
      .def(pybind11::init<int, int, int, uint32_t, py::array_t<uint32_t>,
                          py::array_t<float>, py::array_t<uint32_t>,
                          py::array_t<uint32_t>>())
      .def("decode", &HyperIBLT::Decode, py::arg{}.noconvert());
}
