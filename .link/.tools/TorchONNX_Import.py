import torch.onnx
import torch.utils.model_zoo as zoo
import torchvision
from torch.autograd import Variable

state_dict = zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
model = torchvision.models.alexnet(pretrained=True).cuda()


input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
