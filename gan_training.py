from torch import nn,optim
from torch.nn import functional as F
from torchvision import datasets,models,transforms

def cov(m):
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  
    return fact * m.matmul(mt).squeeze()

def train_gan_model(epochs,P):

	generator=nn.Sequential(nn.Linear(100,100),
                      nn.BatchNorm1d(100),
                      nn.ReLU(),
                      nn.Linear(100,100),
                      nn.BatchNorm1d(100),
                      nn.ReLU(),
                      nn.Linear(100,100),
                      nn.BatchNorm1d(100),
                      nn.ReLU(),
                      nn.Linear(100,100)).cuda()

	discriminator=nn.Sequential(nn.Linear(100,100),
	                      nn.BatchNorm1d(100),
	                      nn.LeakyReLU(),
	                      nn.Linear(100,100),
	                      nn.BatchNorm1d(100),
	                      nn.LeakyReLU(),
	                      nn.Linear(100,100),
	                      nn.BatchNorm1d(100),
	                      nn.LeakyReLU(),
	                      nn.Linear(100,1)).cuda()

	optimizer_discriminator = optim.Adam(discrimator.parameters(),lr=0.0001)
	optimizer_generator = optim.Adam(generator.parameters(),lr=0.0025)

	for epoch in range(epochs):
	    optimizer_discriminator.zero_grad()
	    optimizer_generator.zero_grad()

	    fake_data = generator(torch.rand(5000,100))
	    real_data = torch.tensor(pca.transform(P),dtype=torch.float,device='cuda')
	 
	    discriminator_loss = (discriminator(real)).mean() + (1-discriminator(fake)).mean()

	    intermediate_activations_fake = discriminator[:-1](fake)
	    intermediate_activations_real = discriminator[:-1](real)

	    generator_loss = ((intermediate_activations_fake-intermediate_activations_real)**2).mean() + ((cov(intermediate_activations_fake)-cov(intermediate_activations_real))**2).mean()

	    discriminator_loss.backward()
	    optimizer_discriminator.step()

	    generator_loss.backward()
	    optimizer_generator.step()

		print(discriminator_loss)
		print(generator_loss)