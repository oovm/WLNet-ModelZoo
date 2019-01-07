import cv2
import torch
import wolframclient.serializers as wxf

from src.networks import EdgeGenerator

forward_dict = {}


def make_hook(name):
    def hook(m, input, output):
        forward_dict[name + '_in'] = input[0].detach().numpy()
        forward_dict[name + '_out'] = output[0].detach().numpy()
    return hook


dataset = 'celeba'
data = torch.load('checkpoints/' + dataset + '/EdgeModel_gen.pth', map_location='cpu')
generator = EdgeGenerator().cpu().eval()
generator.load_state_dict(data['generator'])

# 垃圾 torch 不接受 img[:,:,::-1] 或 img[...::-1] 的写法
img = torch.Tensor(cv2.imread('examples/celeba/images/celeba_01.png'))
img = img[:, :, [2, 1, 0]].unsqueeze(0).permute(0, 3, 1, 2).cpu()

layers = list(generator.named_modules())

generator.encoder[0].register_forward_hook(make_hook('encoder.0'))
generator.encoder[1].register_forward_hook(make_hook('encoder.1'))
generator.encoder[2].register_forward_hook(make_hook('encoder.2'))
generator.encoder[3].register_forward_hook(make_hook('encoder.3'))
generator.middle.register_forward_hook(make_hook('middle.0'))
generator.decoder.register_forward_hook(make_hook('decoder.0'))
generator.decoder[-1].register_forward_hook(make_hook('decoder.7'))  # output
out = generator(img)

# npy = {key: value.detach().numpy() for key, value in inter_feature.items()}
wxf.export(forward_dict, 'debug.wxf', target_format='wxf')
wxf.export(out.detach().numpy(), 'out.wxf', target_format='wxf')
