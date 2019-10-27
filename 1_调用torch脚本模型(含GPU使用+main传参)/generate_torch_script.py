import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")

output=traced_script_module(torch.ones(1,3,224,224))
# print(output[0,:5])
# tensor([-0.2855,  0.0275,  0.2519,  0.0518,  0.3241], grad_fn=<SliceBackward>)




