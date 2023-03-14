import torch
import numpy as np

import time

from THHyperIBLT import THHyperIBLT
from fedIBLT import HyperIBLT

def _get_prime(n: int):
    def _is_prime(x):
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True

    while not _is_prime(n):
        n += 1
    return n


def test_encode_decode(numel):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_ROW: int = 3
    NUM_COL: int = _get_prime(int(1.25 * numel/NUM_ROW))

    print(f"size of HyperIBLT : num_row = {NUM_ROW}, num_col = {NUM_COL}")

    NUMEL: int = numel
    NUM_BLOCK: int = 20
    MODULO: int = 114514

    tensor = torch.randn(numel, dtype=torch.float32).to(device)

#    index = torch.nonzero(tensor).view(-1)
#    values = tensor[index]
#    index = index.type(torch.int32)

    index = torch.arange(numel, dtype=torch.int32).to(device)
    values = tensor

    encode_table = THHyperIBLT(NUM_ROW, NUM_COL, NUMEL, NUM_BLOCK)

    print("Start of encode")
    start_time = time.perf_counter()
    encode_table.encode(index, values)
    end_time = time.perf_counter()
    print("End of encode")

    print(f"Encode time = {end_time - start_time}")

    results = np.zeros((NUMEL,), dtype=np.float32)
    key_sum = encode_table.key_sum.numpy().astype(np.uint32)
    value_sum = encode_table.value_sum.numpy().astype(np.float32)
    counter = encode_table.counter.numpy().astype(np.uint32)
    hash_bucket = encode_table.hash_bucket.numpy().astype(np.uint32)

    decode_table = HyperIBLT(
      NUM_ROW,
      NUM_COL,
      NUMEL,
      MODULO,
      key_sum,
      value_sum,
      counter,
      hash_bucket
    )

    start_time = time.perf_counter()
    decode_table.decode(results)
    end_time = time.perf_counter()
    print(f"Decode time = {end_time - start_time}")

    print(f"origin tensor = {tensor}")
    print(f"results = {results}")

if __name__ == "__main__":
    test_encode_decode(10_000_000)     
    
