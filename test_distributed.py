import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class DummyDDPM(nn.Module):
    def __init__(self):
        super(DummyDDPM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        return self.net(x)

def train(rank, world_size):
    setup(rank, world_size)

    # Create a simple DDPM-like model
    model = DummyDDPM().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Dummy dataset and optimizer
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Dummy input tensor
    data = torch.randn(16, 100).to(rank)  # Batch size 16, input size 100
    target = torch.randn(16, 100).to(rank)

    for epoch in range(20):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = criterion(output, target)
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = 2  # Number of GPUs
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
