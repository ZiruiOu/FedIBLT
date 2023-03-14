import torch
import numpy as np

from THHyperIBLT import THHyperIBLT
from fedIBLT import HyperIBLT

def test_encode_decode(numel):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_ROW: int = 3
    NUM_COL: int = int(1.4 * numel)
    NUMEL: int = numel
    NUM_BLOCK: int = 20
    MODULO: int = 114514

    tensor = torch.randn(numel, dtype=torch.float32).to(device)

    index = torch.nonzero(tensor).view(-1)
    values = tensor[index]

    index = index.type(torch.int32)

    encode_table = THHyperIBLT(NUM_ROW, NUM_COL, NUMEL, NUM_BLOCK)

    print("Start of encode")
    encode_table.encode(index, values)
    print("End of encode")

    results = np.zeros((NUMEL,), dtype=np.float32)
    key_sum = encode_table.key_sum.numpy().astype(np.uint32)
    value_sum = encode_table.value_sum.numpy().astype(np.float32)
    counter = encode_table.counter.numpy().astype(np.uint32)
    hash_bucket = encode_table.hash_bucket.numpy().astype(np.uint32)
    print("Start of decode")

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
    print("End of decode")

    decode_table.decode(results)

    print(f"results = {results}")


if __name__ == "__main__":
    test_encode_decode(1_000_000)     
    
