import torch.nn as nn
import numpy as np
import torch
import copy

from RepODCCB import Rep_1d_block

class RepMTCN(nn.Module):
    def __init__(self, in_channels, internal_and_out_chanels, prime_kernel_size=3):
        super(RepMTCN, self).__init__()
        layers = []
        stride = 0
        num_levels = len(internal_and_out_chanels)
        for i in range(num_levels):
            dilation = 2 ** i
            stride += 1
            in_channels = in_channels if i == 0 else internal_and_out_chanels[i-1]
            out_channels = internal_and_out_chanels[i]
            layers += [Rep_1d_block(in_channels, out_channels,
                                    dilation=dilation,prime_kernel_size=prime_kernel_size,stride=stride)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def repmtcn_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

num_inputs = 16 #input channels
num_channels = [32, 64, 64,128]
inputs = torch.randn(100, 16, 100) # 100 16 channels with a tensor of length 100 for each channel sequence

encoder = RepMTCN(num_inputs, num_channels)
encoder.eval()

print(encoder)

print("==================================================Ancestral dividing line==================================================")
#Calculate the result before the encoder conversion
outputs1 = encoder(inputs)
#Convert this encoder
repmtcn_encoder = repmtcn_model_convert(encoder)
print(repmtcn_encoder)
encoder_paramters =  sum(p.numel() for p in encoder.parameters())
repencoder_parameters = sum(p.numel() for p in repmtcn_encoder.parameters())
print("==================================================Ancestral dividing line==================================================")
print("The total paprameters of encoder is:",encoder_paramters)
print("==================================================Ancestral dividing line==================================================")
print("The total paprameters of reptch_encoder is:", repencoder_parameters)
print("==================================================Ancestral dividing line==================================================")
print("The reduction of the parameter after conversion is:",encoder_paramters - repencoder_parameters)
print("==================================================Ancestral dividing line==================================================")
reduced_ratio = (encoder_paramters - repencoder_parameters) / encoder_paramters
print(f"The ratio of the reduced parameter is: {reduced_ratio * 100:.2f}%")
print("==================================================Ancestral dividing line==================================================")

outputs2 = repmtcn_encoder(inputs)

print("The difference between the output before and after conversion is:",end=' ')
print((outputs2 - outputs1).sum().item() ** 2)
print("==================================================Ancestral dividing line==================================================")
print(outputs1.shape)


print(outputs1.max())
print(outputs2.max())

print(outputs1.min())
print(outputs2.min())

