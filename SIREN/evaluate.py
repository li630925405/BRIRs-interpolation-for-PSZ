import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from setting import *
from utils import *
from scipy import ndimage
from module import Siren


def model_res(model_name):
    model = Siren(in_features=2, out_features=1, hidden_features=hidden_features,
                  hidden_layers=5, outermost_linear=True, first_omega_0=omega, hidden_omega_0=omega)
    model.cuda()
    model.load_state_dict(torch.load(model_name + ".pt"))
    # model = torch.load(model_name + ".pt")
    with torch.no_grad():
        upsample_coords = get_mgrid_2d(m, time_len).cuda()
        model_out, _ = model(upsample_coords)
        model_out = model_out.cpu().view(m, time_len).detach().numpy()
    return model_out


res = model_res(name)
# bi_res = ndimage.zoom(rir, zoom=[m/num_x, 1], order=1)

''' sound field visualization '''
# plot_ir(np.array(rir).transpose(), name=folder + "original_"+params)
# plot_ir(res.transpose(), name=folder + model_type + '_' + params)
# plot_ir(truth.transpose(), name=folder + "truth_"+params)
# plot_ir(bi_res.transpose(), name=folder + "bilinear_"+params)


''' save results '''
with open(folder + model_type + '_' + params + '.npy', 'wb') as f:
    np.save(f, res)
# with open(folder + 'bilinear_' + params + '.npy', 'wb') as f:
#     np.save(f, bi_res)
# with open(folder + 'truth_' + params + '.npy', 'wb') as f:
#     np.save(f, truth)


''' print metrics '''
# print(f"bilinear: {time_loss(bi_res, truth):.2f}")
# print(f"{model_type}: {time_loss(res, truth):.2f}")
