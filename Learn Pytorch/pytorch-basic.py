import torch
from torch import nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad = True,
                                                ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x
    
    
model = LinearRegressionModel().to('cpu')

X = torch.randn(10000)
Y = 4 * X + torch.rand(X.size()) / 100

print("X:", X)
print("Y:", Y)

print(model.weights)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(0,100000):
    print(f"==== Epoch {epoch} ====")
    Y_hat = model(X)
    Loss = (Y_hat - Y) ** 2
    print(Loss.mean())
    optimizer.zero_grad()
    Loss.mean().backward()
    optimizer.step()
        

print(model.weights)

print(model(X))