from utils import *
from data import *
from setting import *
from module import Siren
from scipy import ndimage
import scipy.io as scio


if __name__ == "__main__":
    model = Siren(in_features=2, out_features=1, hidden_features=hidden_features,
                  hidden_layers=5, outermost_linear=True, first_omega_0=omega, hidden_omega_0=omega)
    model.cuda()
    steps_til_summary = 50

    optim = torch.optim.AdamW(lr=1e-4, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

    if model_type == "pinn":
        soundfield = Soundfield(truth, collocation, time_len)
        dataloader = DataLoader(soundfield, batch_size=4, num_workers=0)

    sampled_soundfield = Soundfield(rir, num_x, time_len)
    sampled_dataloader = DataLoader(sampled_soundfield, batch_size=400, shuffle=True, num_workers=0)

    sampled_input, sampled_truth = next(iter(sampled_dataloader))
    sampled_truth = sampled_truth.reshape(sampled_truth.shape[1:])
    sampled_truth = sampled_truth.reshape(-1, 1)

    for step in range(training_step + 1):
        sampled_output, _ = model(sampled_input)
        loss = ((sampled_output - sampled_truth) ** 2).mean()

        if model_type == "pinn":
            model_input, _ = next(iter(dataloader))
            model_output, coords = model(model_input)
            pinn_loss = wave_eq(model_output, coords)
            loss = loss + pinn_weight * pinn_loss.pow(2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if not step % 2000:
            scheduler.step()

    # torch.save(model, name + '.pt')
    torch.save(model.state_dict(), name + '.pt')

