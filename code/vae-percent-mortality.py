# Github Repository: https://github.com/rajarshi-mandal/malaria
# 
# Data Processing Notebooks
# - https://www.kaggle.com/code/rm1000/nt-demographic-and-health-surveys-cleaning/
# - https://www.kaggle.com/code/rm1000/nt-earthdata-giovanni-cleaning/
# - https://www.kaggle.com/code/rm1000/nt-ir-mapper-anopheles-cleaning/
# - https://www.kaggle.com/code/rm1000/nt-malaria-atlas-project-cleaning/
# - https://www.kaggle.com/code/rm1000/nt-multiple-indicator-cluster-surveys-cleaning/
# - https://www.kaggle.com/code/rm1000/nt-the-global-health-observatory-cleaning/
# 
# VAE, Differential Equation Modeling, and RL Notebooks
# - https://www.kaggle.com/code/rm1000/nt-vae-percent-mortality/
# - https://www.kaggle.com/code/rm1000/nt-mathematical-modelling/

# ## Data Augmentation


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


ir = pd.read_csv("malaria-data-for-modeling-dynamics/IR Mapper (Anopheles)/insecticide_resistance.csv")
itn_ir = ir[ir['Insecticide class']=='pyrethroid'].drop(['Insecticide tested','Insecticide class'],axis=1)
itn_ir = itn_ir.map(lambda x: x.replace('\x92', "'") if isinstance(x, str) else x)
training = []
countries = list(itn_ir['Country'].value_counts().index)


for country in countries:
    ircountry = itn_ir[itn_ir['Country']==country]
    irconcen = list(ircountry['Concentration (%)'].value_counts().index)
    for ircon in irconcen:
        full = ircountry[ircountry['Concentration (%)']==ircon]
        mort_yr = full.groupby('Year')[['Percent mortality','No. mosquitoes tested']].apply(lambda g: np.average(g['Percent mortality'], weights=g['No. mosquitoes tested']))
        if len(mort_yr) >= 4:
            mort_yr_full = mort_yr.reindex(range(2010, 2018), fill_value=np.nan).tolist()
            training.append([country]+mort_yr_full)


df_dirty = pd.DataFrame(training, columns=['Country',2010,2011,2012,2013,2014,2015,2016,2017])
country_year_means = df_dirty.groupby('Country').transform('mean')
df_dirty = df_dirty.fillna(country_year_means)
df = df_dirty.copy()
year_cols = df_dirty.columns[1:]
row_means = df_dirty[year_cols].mean(axis=1)
col_means = df_dirty[year_cols].mean()
for i, row in df_dirty.iterrows():
    for year in year_cols:
        if pd.isna(df_dirty.at[i, year]):
            df.at[i, year] = (row_means[i] + col_means[year]) / 2


df_old = df.drop('Country',axis=1)
plt.figure(figsize=(6, 4.5))
for i in range(len(df_old)):
    plt.plot(df_old.columns, df_old.iloc[i], alpha=1, linewidth=0.6)
plt.xlabel("Year")
plt.ylabel("Percent Mortality")
plt.title("Before Data Augmentation")
plt.show()


year_cols = [y for y in range(2010, 2018)]
augmented_rows = []
for country,group in df.groupby('Country'):
    if len(group) < 2:
        continue
    rows = group[year_cols].values
    for i, j in itertools.combinations(range(len(rows)), 2):
        row1 = rows[i]
        row2 = rows[j]
        for alpha in [0.25, 0.5, 0.75]:
            interpolated = alpha * row1 + (1 - alpha) * row2
            new_row = [country] + interpolated.tolist()
            augmented_rows.append(new_row)
aug_df = pd.DataFrame(augmented_rows, columns=['Country'] + year_cols)
df = pd.concat([df, aug_df], ignore_index=True).drop('Country',axis=1)


year_cols = df.columns.tolist()
jittered_rows = []
for _, row in df.iterrows():
    original = row.values
    for _ in range(9):
        noise = np.random.normal(loc=0, scale=5, size=original.shape)
        jittered = np.clip(original + noise, 0, 100)
        jittered_rows.append(jittered)
df_jittered = pd.DataFrame(jittered_rows, columns=year_cols)
df = pd.concat([df, df_jittered], ignore_index=True)


plt.figure(figsize=(6, 4.5))
for i in range(len(df)):
    plt.plot(df.columns, df.iloc[i], alpha=0.75, linewidth=0.3)
plt.xlabel("Year")
plt.ylabel("Percent Mortality")
plt.title("After Data Augmentation")
plt.show()


# 0-41 original
# 42 to 101 interpolation
# 102 to 1019 jitter
plt.figure(figsize=(8.5, 2))
sns.heatmap(df.iloc[65:95].T, cmap='Oranges', cbar_kws={'label': 'Percent Mortality'})
plt.title("Interpolation Augmentation Visualized")
plt.ylabel("Year")
plt.xlabel("Row")
plt.xticks([])
plt.show()


# 0-41 original
# 42 to 101 interpolation
# 102 to 1019 jitter
plt.figure(figsize=(8.5, 2))
sns.heatmap(df.iloc[844:874].T, cmap='Oranges', cbar_kws={'label': 'Percent Mortality'})
plt.title("Jitter Augmentation Visualized")
plt.ylabel("Year")
plt.xlabel("Row")
plt.xticks([])
plt.show()


# ## Variational Autoencoder


df = df.sample(frac=1, random_state=42).reset_index(drop=True)
data = torch.tensor(df.values, dtype=torch.float32)
# Normalize to [0,1] for sigmoid output
data = data / 100.0
batch_size = 64
loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self, input_dim=8, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.mu = nn.Linear(8, latent_dim)
        self.logvar = nn.Linear(8, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Sigmoid()  # output in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.mu(x_encoded)
        logvar = self.logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


def vae_loss(reconstructed, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # recon_loss is about 0.1
    # kl div is about 0.03 trains to 0.0003
    return recon_loss + 3*kl_div


def train_vae(latent_space,verbose):
    vae = VAE(input_dim=8, latent_dim=latent_space)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    # Training
    epochs = 50
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(x)
            loss = vae_loss(reconstructed, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
    if not verbose:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
    return vae


def generate_vae(latent_space,vae_model,dpoints,plot):
    # Generating
    vae_model.eval()
    with torch.no_grad():
        # To encourage greater variability in the generated resistance trajectories, we sampled from a widened latent prior distribution 𝑁 ( 0 , 3 ) N(0,3) instead of the standard 𝑁 ( 0 , 1 ) N(0,1). This increased volatility ensures that the reinforcement learning agent is exposed to a broader range of realistic yet challenging scenarios during malaria simulation.
        z = torch.distributions.Normal(0, 3).sample((dpoints, latent_space))
        samples = vae_model.decoder(z) * 100  # scale back to [0,100]
    generated_df = pd.DataFrame(samples.numpy(), columns=df.columns)
    if plot:
        # Visualizing
        plt.figure(figsize=(6, 4.5))
        for i in range(len(generated_df)):
            if dpoints<100:
                plt.plot(generated_df.columns, generated_df.iloc[i], alpha=1, linewidth=0.6)
            else:
                if i<250:
                    plt.plot(generated_df.columns, generated_df.iloc[i], alpha=0.75, linewidth=0.3)
        plt.xlabel("Year")
        plt.ylabel("Percent Mortality")
        plt.title("VAE Samples")
        plt.show()
    return generated_df


# 4 is usually ideal
# Visual comparison to explain latent space
# 1 to 6
for ls in range(1,7):
    vae = train_vae(ls,False)
    generate_vae(ls,vae,42,True)


final_vae = train_vae(3,False)
results = generate_vae(3,final_vae,10000,True)
### DO NOT DELETE
### results.to_csv('random_pm.csv',index=False)


final_vae.eval()
all_mu = []
with torch.no_grad():
    for batch in loader:
        x = batch[0]
        _, mu, _ = final_vae(x)
        all_mu.append(mu)

all_mu = torch.cat(all_mu).squeeze().numpy()
plt.figure(figsize=(5,3.5))
plt.hist(all_mu, bins=10)
plt.title("Latent μ distribution")
plt.show()
