import torch
import torch.nn as nn

import hashlib

MAX_PRIME = (2**61) - 1

__all__ = ['THHyperIBLT']

def _md5_hashbuckets(num_row: int, numel: int, num_col: int):
    pass


class THHyperIBLT(object):
    def __init__(self, num_row, num_col, numel, num_block):
        self.num_row: int = num_row
        self.num_col: int = num_col
        self.numel = numel
        self.num_block = num_block

        self.device = "cuda"

        # FIXME(ozr): no support for torch.uint32
        self.key_sum = torch.zeros((num_row, num_col), dtype=torch.int32)
        self.value_sum = torch.zeros((num_row, num_col), dtype=torch.float32)
        self.counter = torch.zeros((num_row, num_col), dtype=torch.int32)

        # Precompute hash bucket
        rng_state = torch.random.get_rng_state()
        # TODO (ozr): parameter
        torch.random.manual_seed(42)

        self.hash_bucket = torch.randint(
            0, num_col, size=(num_row, numel)
        ).type(torch.int32)

        torch.random.set_rng_state(rng_state)

    def zero(self):
        self.key_sum = torch.zeros((num_row, num_col), dtype=torch.uint32)
        self.value_sum = torch.zeros((num_row, num_col), dtype=torch.float32)
        self.counter = torch.zeros((num_row, num_col), dtype=torch.uint32)

    def encode(self, index: torch.Tensor, values: torch.Tensor):
        self.key_sum = self.key_sum.to(self.device)
        self.value_sum = self.value_sum.to(self.device)
        self.counter = self.counter.to(self.device)
    
        numel_per_block = self.numel // self.num_block
        if self.numel % self.num_block != 0:
            numel_per_block += 1
    
        for row in range(self.num_row):
            for i in range(self.num_block):
                block_start = numel_per_block * i
                block_end = min(block_start + numel_per_block, self.numel)
    
                block_idx = self.hash_bucket[row,block_start:block_end].clone()
                block_idx = block_idx.long().to(self.device)
    
                self.key_sum[row,].scatter_add_(
                    0, block_idx, index[block_start:block_end]
                )
                self.value_sum[row,].scatter_add_(
                    0, block_idx, values[block_start:block_end]
                )
                self.counter[row,].scatter_add_(
                    0, block_idx, torch.ones_like(block_idx,dtype=torch.int32)
                )
    
        self.key_sum = self.key_sum.cpu()
        self.value_sum = self.value_sum.cpu()
        self.counter = self.counter.cpu()
