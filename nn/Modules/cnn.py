import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv1d(in_channels=args.encoder_dim,
                out_channels=args.encoder_dim,
                kernel_size=args.kernel_size,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=args.encoder_dim),
        nn.SiLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    x = x.transpose(1, 2)
    return self.layers(x).transpose(1, 2)

class ConvBlockDecoder(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv1d(in_channels=args.decoder_dim,
                out_channels=args.decoder_dim,
                kernel_size=args.kernel_size,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=args.decoder_dim),
        nn.SiLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    x = x.transpose(1, 2)
    return self.layers(x).transpose(1, 2)

class ResNetLayer(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.conv_layer = nn.Sequential(
        nn.Conv1d(in_channels=args.encoder_dim,
                out_channels=args.encoder_dim,
                kernel_size=3,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=args.encoder_dim),
        nn.SiLU(),
    )
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv_layer(x)+x
        
        
class ResNetBlock(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.layers = nn.Sequential(*[ResNetLayer(args) for _ in range(3)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    return self.layers(x)