import torch
import torch.nn as nn
import onnx

class Flip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        xt = x[:, :, 0::2, :]
        y = torch.flip(xt, [2])
        return y

if __name__ == "__main__":
    model = Flip()
    x = torch.randn(1, 3, 32 , 1)
    y = model(x)
    print(y.shape)

    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "flipnet/flip.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                #                 'output' : {0 : 'batch_size'}}
                                )
    
    onnx_model = onnx.load("flipnet/flip.onnx")
    onnx.checker.check_model(onnx_model)