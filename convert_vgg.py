# convert_vgg.py
import numpy as np
import torch

def convert_vgg_npy_to_pth(npy_path, output_path):
    data = np.load(npy_path, allow_pickle=True, encoding='latin1').item()
    state_dict = {}
    for name, (weight, bias) in data.items():
        if 'fc' in name:
            # FC weight: [in, out] -> [out, in]
            state_dict[name + '.weight'] = torch.FloatTensor(weight).t()
        else:
            # Conv weight: [H, W, in, out] -> [out, in, H, W]
            state_dict[name + '.weight'] = torch.FloatTensor(weight).permute(3,2,0,1)
        state_dict[name + '.bias'] = torch.FloatTensor(bias)
    torch.save(state_dict, output_path)
    print('Saved to', output_path)

if __name__ == '__main__':
    convert_vgg_npy_to_pth('vgg16.npy', 'vgg16.pth')  
