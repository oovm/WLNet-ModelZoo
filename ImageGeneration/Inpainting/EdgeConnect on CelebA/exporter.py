import torch
import wolframclient.serializers as wxf

from src.networks import InpaintGenerator, EdgeGenerator


def edge_model(dataset):
    data = torch.load('checkpoints/' + dataset + '/EdgeModel_gen.pth', map_location='cpu')
    npy = {key: value.numpy() for key, value in data['generator'].items()}
    wxf.export(npy, 'EdgeModel_' + dataset + '.wxf', target_format='wxf')
    generator = EdgeGenerator()
    generator.load_state_dict(data['generator'])
    torch.save(generator, 'EdgeModel_' + dataset + '.pth')


def inpaint_model(dataset):
    data = torch.load('checkpoints/' + dataset + '/InpaintingModel_gen.pth', map_location='cpu')
    npy = {key: value.numpy() for key, value in data['generator'].items()}
    wxf.export(npy, 'InpaintingModel_' + dataset + '.wxf', target_format='wxf')
    generator = InpaintGenerator()
    generator.load_state_dict(data['generator'])
    torch.save(generator, 'InpaintingModel_' + dataset + '.pth')


def export_model(dataset):
    inpaint_model(dataset)
    edge_model(dataset)


for model in ['celeba', 'places2', 'psv']:
    export_model(model)
