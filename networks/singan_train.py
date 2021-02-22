import torch.optim as optim
import torch
import torch.nn as nn
from scipy import signal
def generate_noise(size,num_samp=1,device='cpu',type='uniform', scale=1):
    if type == 'uniform':
        noise = torch.randn(num_samp,size,  device=device)
    return noise.view(num_samp,1,size)
     #   noise = upsampling(noise,size)
    
def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,device='cpu',D_steps=3,G_steps=3,alpha=10):
    netD.to(device)
    netG.to(device)
    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []
    real=torch.from_numpy(reals[len(Gs)]).view(1,1,-1).float()
    seq_length=real.shape[-1]
    optimizerD = optim.Adam(netD.parameters(), lr=0.005, betas=(0.9, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.005, betas=(0.9, 0.999))
    fixed_noise=generate_noise(seq_length,device=device)
    z_opt=torch.full(fixed_noise.shape,0,device=device)
    for epochs in range(200):
        if Gs==[]:
            z_opt=generate_noise(seq_length,device=device)
        noise_ = generate_noise(seq_length,device=device)

    ##Train Discriminator
        for j in range(D_steps):
            netD.zero_grad()
            real=real.to(device)
            output = netD(real)#.to(device)
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()
            if (j==0) & (epochs == 0):
                if (Gs == []):
                    prev = torch.full([1,1,seq_length], 0)
                    in_s = prev
                    z_prev= torch.full([1,1,seq_length], 0)
                    noise_amp=1
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',device=device)
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',device=device)

                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    noise_amp = 0.1*RMSE

            else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',device=device)
            if (Gs == []):
                    noise = noise_
            else:
                    noise = noise_amp*noise_+prev
            noise=noise.to(device)
            prev=prev.to(device)
            fake = netG(noise.detach(),prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = calc_gradient_penalty(netD, real, fake, 0.1,device)
            gradient_penalty.backward()
          #  print(prev)
            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())
        #Train Generator
        for j in range(G_steps):
                netG.zero_grad()
                output = netD(fake)
                errG = -output.mean()
                errG.backward(retain_graph=True)
                if alpha!=0:
                    loss = nn.MSELoss()
                    z_prev=z_prev.to(device)
                    
                    Z_opt = noise_amp*z_opt+z_prev
                    
                    rec_loss = alpha*loss(netG(Z_opt,z_prev),real)
                    rec_loss.backward(retain_graph=True)
                    rec_loss = rec_loss.detach()
                else:
                    Z_opt = z_opt
                    rec_loss = 0

                optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epochs % 50 == 0 :
            print('scale %d:[%d/%d]' % (len(Gs), epochs, 200))
            print("Discriminator Loss: ",abs(errD.item()))
            print("Generator Loss: ",(errG.detach()+rec_loss).item())
    #save_networks(netG,netD,z_opt)        
    return z_opt,in_s,netG ,noise_amp

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,device='cpu'):
    G_z = in_s
    
    if len(Gs) > 0:
        
        if mode == 'rand':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = generate_noise(Z_opt.shape[2],device=device)
                else:
                    z = generate_noise(Z_opt.shape[2],device=device) 
                G_z=G_z.to(device)
                z_in = noise_amp*z+G_z

                G_z = G(z_in.detach(),G_z).cpu().detach().numpy()
                G_z=signal.resample(G_z.squeeze(0).squeeze(0), len(real_next)).reshape(1,1,-1)
                G_z=torch.from_numpy(G_z).to(device)
                count += 1
        if mode == 'rec':
        
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
             #   G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z=G_z.to(device)
                z_in = noise_amp*Z_opt+G_z
                
          
                G_z = G(z_in,G_z).cpu().detach().numpy()
                G_z=signal.resample(G_z.squeeze(0).squeeze(0), len(real_next)).reshape(1,1,-1)
                G_z=torch.from_numpy(G_z).to(device)

    return G_z

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA,device='cpu'):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
