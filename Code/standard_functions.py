import numpy as np
import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import hessian
from torch.func import jacfwd 
from torch.func import vmap 
from torchdiffeq import odeint
from torch.optim.lr_scheduler import _LRScheduler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches


def paste_data(scaled_data, data):

    df = pd.DataFrame(scaled_data,columns=[1,2,3,5,10,15,20,30])
    df.insert(0, 'Currency',data['Currency'])
    df.insert(0, 'Date', data['Date'])

    df_melted = df.melt(id_vars=['Date', 'Currency'], 
                    value_vars=[1, 2, 3, 5, 10, 15, 20, 30], 
                    var_name='Maturity', 
                    value_name='SwapRate')
    return df, df_melted

def calc_rmse(org_data, reconstructed_data):
    currency_rmse = {}
    for currency in org_data["Currency"].unique():

        org = org_data[org_data["Currency"] == currency]
        rec = reconstructed_data[reconstructed_data["Currency"] == currency]
    
        rmse = []

        for i in range(len(org)):
            calc = np.sqrt(np.mean(np.square(org.iloc[i, 2:10] - rec.iloc[i, 2:10]))) * 10000
            rmse.append(calc)
        
        currency_rmse[currency] = np.mean(rmse)

    currency_rmse["Average"] = np.mean([currency_rmse[currency] for currency in currency_rmse.keys()])

    return currency_rmse

currency_color_map = {
    'AUD': 'pink',    
    'CAD': 'grey',  
    'DKK': 'red',   
    'EUR': 'blue',     
    'JPY': 'black',  
    'NOK': 'orange',   
    'SEK': 'purple',    
    'GBP': 'green',   
    'USD': 'brown'  
}

currency_rename_map = {
    'ad': 'AUD',  
    'AD': 'AUD',
    'cd': 'CAD',  
    'CD': 'CAD',
    'dk': 'DKK',  
    'DK': 'DKK',
    'EU': 'EUR',  
    'eu': 'EUR',
    'jy': 'JPY',  
    'JY': 'JPY',
    'nk': 'NOK',  
    'NK': 'NOK',
    'sw': 'SEK',  
    'SK': 'SEK',
    'uk': 'GBP',  
    'BP': 'GBP',
    'US': 'USD',
    'us': 'USD'
}


def train_validation_split(data, test_size):
    xtrain, xval = train_test_split(data, test_size=test_size,random_state=0)
    return xtrain.reset_index(drop=True), xval.reset_index(drop=True)

def check_nan_inf(x, name):
        if torch.isnan(x).any():
            print(f"NaN in {name}!")
        if torch.isinf(x).any():
            print(f"Inf in {name}!")

def detect_nan_inf(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected!")
        if torch.isinf(param).any():
            print(f"Inf detected!")
    
def grad_latent_2factor(res, model):

    calc_grad_G = vmap(jacfwd(model.decoder, argnums=0), in_dims=(0)) 
    grad_G = calc_grad_G(res).squeeze(dim=1) # N*30X3

    gradG_dz   = grad_G[:, :2] # N*30X2
    dG_dm      = grad_G[:, 2:] # N*30X1

    return gradG_dz, dG_dm

def hess_latent_2factor(res, model):
    
    calc_hess_G = vmap(jacfwd(jacfwd(model.decoder, argnums=0), argnums=0), in_dims=(0))
    hess_G = calc_hess_G(res).squeeze(dim=1)

    hessG_dz = hess_G[:,:2,:2]
    
    return hessG_dz

def alpha_fct_2factor(res, model, mu, sigma, G):
    gradG_dz, dG_dm = grad_latent_2factor(res, model)
    hessG_dz = hess_latent_2factor(res, model)

    part1 = -dG_dm

    part2 = torch.matmul(gradG_dz.unsqueeze(1), mu.unsqueeze(2))#.detach()
    part2 = part2.squeeze(-1)

    part0 = torch.matmul(sigma.transpose(1, 2), torch.matmul(hessG_dz, sigma))#.detach()

    part3 = 0.5 * torch.einsum('bii->b', part0).unsqueeze(-1)

    return (part1 + part2 + part3) / G

def beta_fct(r,G):
    return r/G

def gamma_fct_2factor(res, model, sigma):
    gradG_dz, _ = grad_latent_2factor(res, model)

    grad_grad = torch.matmul(gradG_dz.unsqueeze(-1), gradG_dz.unsqueeze(-1).transpose(1, 2))
    part0 = torch.matmul(sigma.transpose(1, 2), grad_grad.matmul(sigma))

    return 0.5 * torch.einsum('bii->b', part0).unsqueeze(-1)

def build_sigma_matrix_2factor(sigma_1, sigma_2, rho):
    sigma_matrix = torch.zeros((len(sigma_1), 2, 2))  
    sigma_matrix[:, 0, 0] = torch.exp(sigma_1.squeeze(-1))
    sigma_matrix[:, 0, 1] = 0  
    sigma_matrix[:, 1, 0] = torch.tanh(rho.squeeze(-1)) * torch.exp(sigma_2.squeeze(-1))
    sigma_matrix[:, 1, 1] = torch.sqrt(1 - torch.tanh(rho.squeeze(-1))**2) * torch.exp(sigma_2.squeeze(-1))

    return sigma_matrix.repeat_interleave(30, dim=0)

class ODESystem_constant(torch.nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(ODESystem_constant, self).__init__()
        self.alpha = alpha  # N x 30
        self.beta = beta    # N x 30
        self.gamma = gamma  # N x 30

    def forward(self, t, x):
        N = x.shape[0]
        x_reshaped = x.view(N, 2)
        
        x1 = x_reshaped[..., 1] 

        dx0dt = self.gamma * (x1**2)          # N x 30
        dx1dt = self.alpha * x1 + self.beta   # N x 30

        dxdt_reshaped = torch.stack([dx0dt, dx1dt], dim=-1)
        dxdt = dxdt_reshaped.view(N, 2)

        return dxdt

def reshape_wide(x, length):
    x = x.reshape(length, 30)
    return x

def arb_equation_2factor(G, sigma_1, sigma_2, rho , mu, r, encoded_mat, model, dA, dB, B):
    N = len(G)
    
    r_long = r.repeat_interleave(30, dim=0) 
    G_long = G.reshape(N*30, 1)
    dA_long = dA.reshape(N*30, 1)
    dB_long = dB.reshape(N*30, 1)
    B_long = B.reshape(N*30, 1)
    mu_long = mu.repeat_interleave(30, dim=0)      
    grad_z, dy_dm   = grad_latent_2factor(encoded_mat, model)  
    
    
    grad_zmu = torch.matmul(grad_z.unsqueeze(1), mu_long.unsqueeze(2)).squeeze(-1) 
    
    hess_z  = hess_latent_2factor(encoded_mat, model) 
    sigma_long = build_sigma_matrix_2factor(sigma_1, sigma_2, rho)
    sigma_hess_sigma = torch.matmul(sigma_long.transpose(1, 2), torch.matmul(hess_z, sigma_long)) 
    trace_hess = 0.5 * torch.einsum('bii->b', sigma_hess_sigma).unsqueeze(-1)
    
    grad_grad = torch.matmul(grad_z.unsqueeze(-1), grad_z.unsqueeze(-1).transpose(1, 2))
    sigma_grad_grad_sigma = torch.matmul(sigma_long.transpose(1, 2), torch.matmul(grad_grad, sigma_long)) 
    trace_grad_grad = 0.5 * torch.einsum('bii->b', sigma_grad_grad_sigma).unsqueeze(-1)
    
    final_term = -r_long - dA_long + G_long*dB_long + B_long*(dy_dm - grad_zmu - trace_hess) + (B_long**2)*trace_grad_grad   
    
    return final_term

def solve_ODE_constant(alpha, beta, gamma, maturities):
    
    A = torch.zeros_like(alpha, requires_grad=True)
    B = torch.zeros_like(alpha, requires_grad= True) 

    alpha_T = alpha * maturities          # (N,30)
    e1 = torch.expm1(alpha_T)

    B_nonzero = (beta / alpha) * e1
    B_zero = beta * maturities
    B = torch.where(B_nonzero.abs() > 1e-11, B_nonzero, B_zero) 

    A = gamma*B**2*maturities

    return A, B

class LR_Scheduler(_LRScheduler):
    def __init__(self, optimizer, percentage = 0.9, interval = 25, last_epoch = -1):

       self.percentage = percentage
       self.interval = interval
       self.decay_factor = 1.0 

       super().__init__(optimizer, last_epoch)
       
    def get_lr(self):
        if self.last_epoch % self.interval == 0 and self.last_epoch > 0:
            self.decay_factor *= self.percentage
        
        return [base_lr * self.decay_factor for base_lr in self.base_lrs]




def train_ae(fct, train_data, val_data, swap_mats, num_epochs, data_loader, model, criterion, optimizer, lr_sched = None):
    swap_mats0 = [i-1 for i in swap_mats]

    arb_losses_list = []
    losses_list = []
    vallosses_list = []
    std_alp_epoch = 0
    std_bet_epoch = 0
    std_gam_epoch = 0
    std_A_epoch = 0
    std_B_epoch = 0

    for epoch in range(num_epochs):
        arb_loss_epoch = 0
        #loss_epoch = 0

        for batch in data_loader:

            N = len(batch)

            G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat = model(batch)

            reconstructed, p, arb_l, alpha, beta, gamma, A, B, std_alp, std_bet, std_gam, std_A, std_B  = fct(G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat, N, model)
            #print(reconstructed.shape)
            s_final = reconstructed[:, swap_mats0]
            #print(s_final)
            loss = criterion(s_final, batch)
        
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            max_norm = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            nan_detected = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected: {epoch}, resetting grad to zero ")
                    param.grad.zero_() 
                    nan_detected = True

            if not nan_detected:
                optimizer.step()
            else:
                print(f"Skipping optimizer ")

        # summing losses in epoch
            arb_loss_epoch += torch.sum(arb_l)
            #loss_epoch += loss
            std_alp_epoch += std_alp
            std_bet_epoch += std_bet
            std_gam_epoch += std_gam
            std_A_epoch += std_A
            std_B_epoch += std_B

            detect_nan_inf(model)

        if lr_sched is not None:
            lr_sched.step()

        # evaluate validation efter epoch is run 
        N = len(val_data)    
        G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat = model(val_data)
        reconstructed, p, arb_l, alpha, beta, gamma, A, B, std_alp, std_bet, std_gam, std_A, std_B  = fct(G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat, N, model)
        s_final = reconstructed[:, swap_mats0]
        loss_val = criterion(s_final, val_data)

        #
        N = len(train_data)    
        G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat = model(train_data)
        reconstructed, p, arb_l, alpha, beta, gamma, A, B, std_alp, std_bet, std_gam, std_A, std_B  = fct(G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat, N, model)
        s_final = reconstructed[:, swap_mats0]
        loss_train = criterion(s_final, train_data)

        # append relevant data
        vallosses_list.append(loss_val.item())
        arb_losses_list.append(arb_loss_epoch.item())
        losses_list.append(loss_train.item())
        current_lr = optimizer.param_groups[0]["lr"] 
    
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], LR:{current_lr:.8f}, Loss: {loss_train.item():.8f}, Arb:{arb_loss_epoch} ")


    std_alp_epoch = std_alp_epoch/(len(train_data)*num_epochs)
    std_bet_epoch = std_bet_epoch/(len(train_data)*num_epochs)
    std_gam_epoch = std_gam_epoch/(len(train_data)*num_epochs)
    std_A_epoch = std_A_epoch/(len(train_data)*num_epochs)
    std_B_epoch = std_B_epoch/(len(train_data)*num_epochs)
    
    return std_alp_epoch, std_bet_epoch, std_gam_epoch, std_A_epoch, std_B_epoch, arb_losses_list, losses_list, vallosses_list


def reconstruct_outputs(data_tensor, data_scaled, data, fct, model):
    with torch.no_grad():
    
        N = len(data_tensor)
        G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat, encoded = model(data_tensor)
        reconstructed_data, p, arb_l, alpha, beta, gamma, A, B, _, _, _, _, _ = fct(G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat, N, model, plot=True)

    swap_mats = [1, 2, 3, 5, 10, 15, 20, 30]
    swap_mats0 = [i-1 for i in swap_mats]
    X_reconstructed = reconstructed_data[:,swap_mats0]

    df_original = pd.DataFrame(data_scaled,columns=[1,2,3,5,10,15,20,30])
    df_original.insert(0, 'Currency',data['Currency'])
    df_original.insert(0, 'Date', data['Date'])

    # Reshape to long for encoded data 
    df_org_melted = df_original.melt(id_vars=['Date', 'Currency'], 
                    value_vars=[1, 2, 3, 5, 10, 15, 20, 30], 
                    var_name='Maturity', 
                    value_name='SwapRate')

    df_reconstruct = pd.DataFrame(X_reconstructed,columns=[1,2,3,5,10,15,20,30])
    df_reconstruct.insert(0, 'Currency', data['Currency'])
    df_reconstruct.insert(0, 'Date', data['Date'])

    df_rec_melted = df_reconstruct.melt(id_vars=['Date', 'Currency'], 
                    value_vars=[1, 2, 3, 5, 10, 15, 20, 30], 
                    var_name='Maturity', 
                    value_name='SwapRate')
    
    currency_rmse = {}

    for currency in df_original["Currency"].unique():
    
        org = df_original[df_original["Currency"] == currency]
        rec = df_reconstruct[df_reconstruct["Currency"] == currency]
    
        rmse = []
        for i in range(len(org)):
            calc = np.sqrt(np.mean(np.square(org.iloc[i, 2:10] - rec.iloc[i, 2:10]))) * 10000
            rmse.append(calc)
    
        currency_rmse[currency] = np.mean(rmse)
    currency_rmse["Average"]   = np.mean([currency_rmse[currency] for currency in currency_rmse.keys()])
    print("Loss RMSE")
    for key in currency_rmse:
        print(f"{key} : {currency_rmse[key]}")

    arb_l = arb_l.reshape(len(data), 30)


    df_arb = pd.DataFrame(arb_l,columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    df_arb.insert(0, 'Currency', data['Currency'])
    df_arb.insert(0, 'Date', data['Date'])

    currency_rmse_arb = {}

    for currency in df_arb["Currency"].unique():
        org_arb = df_arb[df_arb["Currency"] == currency]

        rmse_arb = []
        calc_arb = (np.mean(np.abs(org_arb.iloc[:, 2:]))) * 10000
        rmse_arb.append(calc_arb)
        
        currency_rmse_arb[currency] = np.mean(rmse_arb)

    currency_rmse_arb["Average"] = np.mean([currency_rmse_arb[currency] for currency in currency_rmse_arb.keys()])

    print("Arb RMSE")
    for key in currency_rmse_arb:
        print(f"{key} : {currency_rmse_arb[key]}")

    return df_rec_melted, df_org_melted, encoded


def reconstruct_outputs_3factor(data_tensor, data_scaled, data, fct, model):
    with torch.no_grad():
    
        N = len(data_tensor)
        G_output, sigma_1, sigma_2, sigma_3, rho12, rho13, rho23, mu, r, encoded_mat, encoded = model(data_tensor)
        reconstructed_data, p, arb_l, alpha, beta, gamma, A, B, _, _, _, _, _ = fct(G_output, sigma_1, sigma_2, sigma_3, rho12, rho13, rho23, mu, r, encoded_mat, N, model, plot=True)

    swap_mats = [1, 2, 3, 5, 10, 15, 20, 30]
    swap_mats0 = [i-1 for i in swap_mats]
    X_reconstructed = reconstructed_data[:,swap_mats0]

    # currency = data['Currency']
    # unique_currencies = np.unique(currency)
    # colors = plt.cm.tab10(np.linspace(0, 1, len(unique_currencies)))
    # seen_curr = set()
    # for i in range(A.size(0)):
    #     color_idx = np.where(unique_currencies == currency[i])[0][0]

    #     curr = currency[i] if currency[i] not in seen_curr else ""  # Avoid duplicate legend labels
    #     seen_curr.add(currency[i])

    #     plt.plot(range(1, 31), A[i].numpy(), color=colors[color_idx], alpha=0.4, label=curr)

    # if len(unique_currencies) > 1:
    #     plt.legend(title="Currency")
    # plt.show()

    # seen_curr = set()
    # for i in range(B.size(0)):
    #     color_idx = np.where(unique_currencies == currency[i])[0][0]

    #     curr = currency[i] if currency[i] not in seen_curr else ""  # Avoid duplicate legend labels
    #     seen_curr.add(currency[i])

    #     plt.plot(range(1, 31), B[i].numpy(), color=colors[color_idx], alpha=0.4, label=curr)

    # if len(unique_currencies) > 1:
    #     plt.legend(title="Currency")
    # plt.show()

    df_original = pd.DataFrame(data_scaled,columns=[1,2,3,5,10,15,20,30])
    df_original.insert(0, 'Currency',data['Currency'])
    df_original.insert(0, 'Date', data['Date'])

    df_org_melted = df_original.melt(id_vars=['Date', 'Currency'], 
                    value_vars=[1, 2, 3, 5, 10, 15, 20, 30], 
                    var_name='Maturity', 
                    value_name='SwapRate')

    df_reconstruct = pd.DataFrame(X_reconstructed,columns=[1,2,3,5,10,15,20,30])
    df_reconstruct.insert(0, 'Currency', data['Currency'])
    df_reconstruct.insert(0, 'Date', data['Date'])

    df_rec_melted = df_reconstruct.melt(id_vars=['Date', 'Currency'], 
                    value_vars=[1, 2, 3, 5, 10, 15, 20, 30], 
                    var_name='Maturity', 
                    value_name='SwapRate')
    
    currency_rmse = {}

    for currency in df_original["Currency"].unique():
    
        org = df_original[df_original["Currency"] == currency]
        rec = df_reconstruct[df_reconstruct["Currency"] == currency]
    
        rmse = []
        for i in range(len(org)):
            calc = np.sqrt(np.mean(np.square(org.iloc[i, 2:10] - rec.iloc[i, 2:10]))) * 10000
            rmse.append(calc)
    
        currency_rmse[currency] = np.mean(rmse)
    currency_rmse["Average"]   = np.mean([currency_rmse[currency] for currency in currency_rmse.keys()])
    
    print("Loss RMSE")
    for key in currency_rmse:
        print(f"{key} : {currency_rmse[key]}")

    arb_l = arb_l.reshape(len(data), 30)


    df_arb = pd.DataFrame(arb_l,columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    df_arb.insert(0, 'Currency', data['Currency'])
    df_arb.insert(0, 'Date', data['Date'])

    currency_rmse_arb = {}

    for currency in df_arb["Currency"].unique():
        org_arb = df_arb[df_arb["Currency"] == currency]

        rmse_arb = []
        calc_arb = (np.mean(np.abs(org_arb.iloc[:, 2:]))) * 10000
        rmse_arb.append(calc_arb)
        
        currency_rmse_arb[currency] = np.mean(rmse_arb)

    currency_rmse_arb["Average"] = np.mean([currency_rmse_arb[currency] for currency in currency_rmse_arb.keys()])

    print("Arb RMSE")
    for key in currency_rmse_arb:
        print(f"{key} : {currency_rmse_arb[key]}")

    df4 = pd.DataFrame(arb_l,columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    for i in df4.columns:            
        df4[i] = df4[i].astype(float)
    
    df4.insert(0, 'Currency', data['Currency'])
    df2 = df4.groupby('Currency').mean()*10000  

    df2_transposed = df2.T
    df2_transposed = df2_transposed.round(5)

    return df_rec_melted, df_org_melted, encoded, df2_transposed

def parameter_data(parameter, data):
    df = pd.DataFrame(parameter)
    df[0] = df[0].astype(float)
    df.insert(0, 'Currency', data['Currency'])
    df.insert(0, 'Date', data['Date'])  

    df['Currency'] = df['Currency'].replace(currency_rename_map)
    df["Date"] = pd.to_datetime(df["Date"]) 
    df.columns = ['Date', 'Currency', '']

    return df

def plot_parameters_2f(data_tensor, data_scaled, data, fct, model):
    with torch.no_grad():
    
        N = len(data_tensor)
        G_output, sigma_1, sigma_2, rho, mu, r, encoded_mat = model(data_tensor)

    df_r = parameter_data(r, data)
    df_sigma1 = parameter_data(torch.exp(sigma_1), data)
    df_sigma2 = parameter_data(torch.exp(sigma_2), data)
    df_mu1 = parameter_data(mu[:,0], data)
    df_mu2 = parameter_data(mu[:,1], data)
    df_rho12 = parameter_data(torch.tanh(rho), data)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_r, x="Date", y="", hue="Currency", palette=currency_color_map, ax=ax, legend=False)
    ax.set_title("Plot of r across currencies")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    sns.lineplot(data=df_rho12, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[0])
    axes[0].set_title('Plot for rho12')
    axes[0].get_legend().remove()

    sns.lineplot(data=df_sigma1, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[1]) 
    axes[1].set_title('Plot for sigma_1')
    axes[1].get_legend().remove()

    sns.lineplot(data=df_sigma2, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[2]) 
    axes[2].set_title('Plot for sigma_2')
    axes[2].get_legend().remove()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(25, 10))

    sns.lineplot(data=df_mu1, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[0]) 
    axes[0].set_title('Plot for mu1')
    axes[0].get_legend().remove()

    sns.lineplot(data=df_mu2, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[1]) 
    axes[1].set_title('Plot for mu2')
    axes[1].get_legend().remove()

    plt.show()

def plot_parameters_2f_2nd(data_tensor, data_scaled, data, fct, model):
    with torch.no_grad():
    
        N = len(data_tensor)
        r, y, p, sigma_1, sigma_2, rho, mu, encoded, encoded_mat, reconstructed, mat = model(data_tensor)

    df_r = parameter_data(r, data)
    df_sigma1 = parameter_data(torch.exp(sigma_1), data)
    df_sigma2 = parameter_data(torch.exp(sigma_2), data)
    df_mu1 = parameter_data(mu[:,0], data)
    df_mu2 = parameter_data(mu[:,1], data)
    df_rho12 = parameter_data(torch.tanh(rho), data)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_r, x="Date", y="", hue="Currency", palette=currency_color_map, ax=ax, legend=False)
    #ax.set_title("Plot of r across currencies")
    ax.set_ylabel("r")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    sns.lineplot(data=df_rho12, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[0])
    axes[0].set_title('Plot for rho12')
    axes[0].get_legend().remove()

    sns.lineplot(data=df_sigma1, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[1]) 
    axes[1].set_title('Plot for sigma_1')
    axes[1].get_legend().remove()

    sns.lineplot(data=df_sigma2, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[2]) 
    axes[2].set_title('Plot for sigma_2')
    axes[2].get_legend().remove()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(25, 10))

    sns.lineplot(data=df_mu1, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[0]) 
    axes[0].set_title('Plot for mu1')
    axes[0].get_legend().remove()

    sns.lineplot(data=df_mu2, x="Date", y="", hue="Currency", palette=currency_color_map, ax=axes[1]) 
    axes[1].set_title('Plot for mu2')
    axes[1].get_legend().remove()

    plt.show()


def plot_parameters_3f(data_tensor, data_scaled, data, fct, model):
    with torch.no_grad():
    
        N = len(data_tensor)
        G_output, sigma_1, sigma_2, sigma_3, rho12, rho13, rho23, mu, r, encoded_mat, encoded = model(data_tensor)

    df_r = parameter_data(r, data)
    df_sigma1 = parameter_data(torch.exp(sigma_1), data)
    df_sigma2 = parameter_data(torch.exp(sigma_2), data)
    df_sigma3 = parameter_data(torch.exp(sigma_3), data)
    df_mu1 = parameter_data(mu[:,0], data)
    df_mu2 = parameter_data(mu[:,1], data)
    df_mu3 = parameter_data(mu[:,2], data)
    df_rho12 = parameter_data(torch.tanh(rho12), data)
    df_rho13 = parameter_data(torch.tanh(rho13), data)
    df_rho23 = parameter_data(torch.tanh(rho23), data)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_r, x="Date", y="", hue="Currency", palette=currency_color_map, legend=False)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    sns.lineplot(data=df_sigma1, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[0], legend=False) 
    axes[0].set_title('Plot for sigma_1')

    sns.lineplot(data=df_sigma2, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[1], legend=False) 
    axes[1].set_title('Plot for sigma_2')

    sns.lineplot(data=df_sigma3, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[2], legend=False)
    axes[2].set_title('Plot for sigma_3')


    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))


    sns.lineplot(data=df_mu1, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[0], legend=False) 
    axes[0].set_title('Plot for mu1')

    sns.lineplot(data=df_mu2, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[1], legend=False) 
    axes[1].set_title('Plot for mu2')

    sns.lineplot(data=df_mu3, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[2], legend=False) 
    axes[2].set_title('Plot for mu3')

    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    sns.lineplot(data=df_rho12, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[0], legend=False)
    axes[0].set_title('Plot for rho12')

    sns.lineplot(data=df_rho13, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[1], legend=False) 
    axes[1].set_title('Plot for rho13')

    sns.lineplot(data=df_rho23, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[2], legend=False)
    axes[2].set_title('Plot for rho23')

    plt.show()

def plot_parameters_3f_2nd(data_tensor, data_scaled, data, fct, model):
    with torch.no_grad():
    
        N = len(data_tensor)
        r, y, p, sigma_1, sigma_2, sigma_3, rho12, rho13, rho23, mu, encoded, encoded_mat, reconstructed, mat = model(data_tensor)

    df_r = parameter_data(r, data)
    df_sigma1 = parameter_data(torch.exp(sigma_1), data)
    df_sigma2 = parameter_data(torch.exp(sigma_2), data)
    df_sigma3 = parameter_data(torch.exp(sigma_3), data)
    df_mu1 = parameter_data(mu[:,0], data)
    df_mu2 = parameter_data(mu[:,1], data)
    df_mu3 = parameter_data(mu[:,2], data)
    df_rho12 = parameter_data(torch.tanh(rho12), data)
    df_rho13 = parameter_data(torch.tanh(rho13), data)
    df_rho23 = parameter_data(torch.tanh(rho23), data)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_r, x="Date", y="", hue="Currency", palette=currency_color_map, legend=False)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    sns.lineplot(data=df_sigma1, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[0], legend=False) 
    axes[0].set_title('Plot for sigma_1')

    sns.lineplot(data=df_sigma2, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[1], legend=False) 
    axes[1].set_title('Plot for sigma_2')

    sns.lineplot(data=df_sigma3, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[2], legend=False)
    axes[2].set_title('Plot for sigma_3')


    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))


    sns.lineplot(data=df_mu1, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[0], legend=False) 
    axes[0].set_title('Plot for mu1')

    sns.lineplot(data=df_mu2, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[1], legend=False) 
    axes[1].set_title('Plot for mu2')

    sns.lineplot(data=df_mu3, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[2], legend=False) 
    axes[2].set_title('Plot for mu3')

    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    sns.lineplot(data=df_rho12, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[0], legend=False)
    axes[0].set_title('Plot for rho12')

    sns.lineplot(data=df_rho13, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[1], legend=False) 
    axes[1].set_title('Plot for rho13')

    sns.lineplot(data=df_rho23, x="Date", y="", hue="Currency",palette=currency_color_map, ax=axes[2], legend=False)
    axes[2].set_title('Plot for rho23')

    plt.show()

class centered_softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x)) - 0.5
    
def solve_ODE_forward_centraldiff(alpha, beta, gamma):
    
    B_values = []  
    A_values = [] 

    for t in range(30):
        

        if t == 0:
            B_prev = 0
            A_prev = 0

            a = (alpha[:, t])
            b = (beta[:, t])
            g = (gamma[:, t])

        else:
            a = (alpha[:, t-1] + alpha[:, t])/2
            b = (beta[:, t-1] + beta[:, t])/2
            g = (gamma[:, t-1] + gamma[:, t])/2

        a = torch.clamp(a, min=-80, max=80)

        B_nonzero = torch.exp(a) * B_prev + (b / a) * torch.expm1(a)
        B_zero = B_prev + b
            
        B_t = torch.where(torch.abs(a) > 1e-11, B_nonzero, B_zero) 
        A_t = A_prev + g * (B_prev**2)  

        A_t = torch.clamp(A_t, min=-1000, max=1000)
        B_t = torch.clamp(B_t, min=-1000, max=1000)

        B_values.append(B_t)
        A_values.append(A_t)

        B_prev = B_t
        A_prev = A_t

    B = torch.stack(B_values, dim=1)
    A = torch.stack(A_values, dim=1)

    return A, B

def solve_ODE_hybrid(alpha, beta, gamma, maturities):
    
    B = torch.zeros_like(alpha, requires_grad= True) 

    alpha_T = alpha * maturities          # (N,30)
    e1 = torch.expm1(alpha_T)

    B_nonzero = (beta / alpha) * e1
    B_zero = beta * maturities

    B = torch.where(B_nonzero.abs() > 1e-11, B_nonzero, B_zero) 

    A_values = [] 

    for t in range(30):
        if t == 0:
            A_prev = 0
            g = (gamma[:, t])

        else:
            g = (gamma[:, t-1] + gamma[:, t])/2

        B_prev = B[:, t]
        A_t = A_prev + g * (B_prev**2)  

        A_t = torch.clamp(A_t, min=-200, max=200)

        A_values.append(A_t)
        A_prev = A_t

    A = torch.stack(A_values, dim=1)

    return A, B

# def solve_ODE_part_constant(alpha, beta, gamma, maturities):
    
#     A = torch.zeros_like(alpha, requires_grad=True)
#     B = torch.zeros_like(alpha, requires_grad= True) 

#     alpha_T = alpha * maturities          # (N,30)
#     e1 = torch.expm1(alpha_T)
#     e2 = torch.expm1(2*alpha_T) 

#     B_nonzero = (beta / alpha) * e1
#     B_zero = beta * maturities

#     B = torch.where(B_nonzero.abs() > 1e-11, B_nonzero, B_zero) 

#     term1 = e2 / (2.0 * alpha)       # Shape: (N, 30)
#     term2 = (2.0 * e1) / alpha       # Shape: (N, 30)
#     A_nonzero = gamma * (beta ** 2) / (alpha ** 2) * (term1 - term2 + maturities)
#     A_zero = gamma*(beta ** 2)*(maturities ** 3)/3

#     A = torch.where(A_nonzero.abs() > 1e-11, A_nonzero, A_zero) 

    # return A, B

def grad_latent_3factor(res, model):

    calc_grad_G = vmap(jacfwd(model.decoder, argnums=0), in_dims=(0)) 
    grad_G = calc_grad_G(res).squeeze(dim=1) # N*30X4

    gradG_dz   = grad_G[:, :3] # N*30X3
    dG_dm      = grad_G[:, 3:] # N*30X1

    return gradG_dz, dG_dm

def hess_latent_3factor(res, model):
    
    calc_hess_G = vmap(jacfwd(jacfwd(model.decoder, argnums=0), argnums=0), in_dims=(0))
    hess_G = calc_hess_G(res).squeeze(dim=1)

    hessG_dz = hess_G[:,:3,:3]
    
    return hessG_dz

def alpha_fct_3factor(res, model, mu, sigma, G):
    gradG_dz, dG_dm = grad_latent_3factor(res, model)
    hessG_dz = hess_latent_3factor(res, model)

    part1 = -dG_dm

    part2 = torch.matmul(gradG_dz.unsqueeze(1), mu.unsqueeze(2))#.detach()
    part2 = part2.squeeze(-1)

    part0 = torch.matmul(sigma.transpose(1, 2), torch.matmul(hessG_dz, sigma))#.detach()

    part3 = 0.5 * torch.einsum('bii->b', part0).unsqueeze(-1)

    return (part1 + part2 + part3) / G

def gamma_fct_3factor(res, model, sigma):
    gradG_dz, _ = grad_latent_3factor(res, model)

    grad_grad = torch.matmul(gradG_dz.unsqueeze(-1), gradG_dz.unsqueeze(-1).transpose(1, 2))
    part0 = torch.matmul(sigma.transpose(1, 2), grad_grad.matmul(sigma))

    return 0.5 * torch.einsum('bii->b', part0).unsqueeze(-1)

def build_sigma_matrix_3factor(sigma_1, sigma_2, sigma_3, rho12, rho13, rho23):
    sigma_matrix = torch.zeros((len(sigma_1), 3, 3))  

    calc_term = (torch.tanh(rho23.squeeze(-1)))-torch.tanh(rho12.squeeze(-1))*torch.tanh(rho13.squeeze(-1))\
        /(torch.sqrt(1 - torch.tanh(rho12.squeeze(-1))**2) + 1e-4)

    sigma_matrix[:, 0, 0] = torch.exp(sigma_1.squeeze(-1))
    sigma_matrix[:, 0, 1] = sigma_matrix[:, 0, 2] = sigma_matrix[:, 1, 2] = 0
    sigma_matrix[:, 1, 0] = torch.tanh(rho12.squeeze(-1)) * torch.exp(sigma_2.squeeze(-1))
    sigma_matrix[:, 1, 1] = torch.sqrt(1 - torch.tanh(rho12.squeeze(-1))**2) * torch.exp(sigma_2.squeeze(-1))
    sigma_matrix[:, 2, 0] = torch.tanh(rho13.squeeze(-1)) * torch.exp(sigma_3.squeeze(-1))
    sigma_matrix[:, 2, 1] = calc_term * torch.exp(sigma_3.squeeze(-1))
    sigma_matrix[:, 2, 2] = torch.sqrt(torch.clamp(1 - torch.tanh(rho13.squeeze(-1))**2 - calc_term**2, min=1e-4)) * torch.exp(sigma_3.squeeze(-1))
    
    return sigma_matrix.repeat_interleave(30, dim=0)

def arb_equation_3factor(G, sigma_1, sigma_2, sigma_3, rho12, rho13, rho23, mu, r, encoded_mat, model, dA, dB, B):
    N = len(G)
    
    r_long = r.repeat_interleave(30, dim=0) 
    G_long = G.reshape(N*30, 1)
    dA_long = dA.reshape(N*30, 1)
    dB_long = dB.reshape(N*30, 1)
    B_long = B.reshape(N*30, 1)
    mu_long = mu.repeat_interleave(30, dim=0)      
    grad_z, dy_dm   = grad_latent_3factor(encoded_mat, model)  
    
    
    grad_zmu = torch.matmul(grad_z.unsqueeze(1), mu_long.unsqueeze(2)).squeeze(-1) 
    
    hess_z  = hess_latent_3factor(encoded_mat, model) 
    sigma_long = build_sigma_matrix_3factor(sigma_1, sigma_2, sigma_3, rho12, rho13, rho23)
    sigma_hess_sigma = torch.matmul(sigma_long.transpose(1, 2), torch.matmul(hess_z, sigma_long)) 
    trace_hess = 0.5 * torch.einsum('bii->b', sigma_hess_sigma).unsqueeze(-1)
    
    grad_grad = torch.matmul(grad_z.unsqueeze(-1), grad_z.unsqueeze(-1).transpose(1, 2))
    sigma_grad_grad_sigma = torch.matmul(sigma_long.transpose(1, 2), torch.matmul(grad_grad, sigma_long)) 
    trace_grad_grad = 0.5 * torch.einsum('bii->b', sigma_grad_grad_sigma).unsqueeze(-1)
    
    final_term = -r_long - dA_long + G_long*dB_long + B_long*(dy_dm - grad_zmu - trace_hess) + (B_long**2)*trace_grad_grad   
    
    return final_term


def train_ae3(fct, train_data, val_data, swap_mats, num_epochs, data_loader, model, criterion, optimizer, lr_sched = None):
    swap_mats0 = [i-1 for i in swap_mats]

    arb_losses_list = []
    losses_list = []
    vallosses_list = []
    std_alp_epoch = 0
    std_bet_epoch = 0
    std_gam_epoch = 0
    std_A_epoch = 0
    std_B_epoch = 0

    for epoch in range(num_epochs):
        arb_loss_epoch = 0
        # loss_epoch = 0
        i = 0
        for batch in data_loader:

            i += 1

            N = len(batch)

            G_output, sigma_1, sigma_2, sigma3, rho12, rho13, rho23, mu, r, encoded_mat, encoded = model(batch)

            reconstructed, p, arb_l, alpha, beta, gamma, A, B, std_alp, std_bet, std_gam, std_A, std_B  = fct(G_output, sigma_1, sigma_2, sigma3, rho12, rho13, rho23, mu, r, encoded_mat, N, model)
            #print(reconstructed.shape)
            s_final = reconstructed[:, swap_mats0]
            #print(s_final)
            loss = criterion(s_final, batch)
        
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            max_norm = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            nan_detected = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients at epoch {epoch}, resetting to zero.")
                    param.grad.zero_() 
                    nan_detected = True

            if not nan_detected:
                optimizer.step()
            else:
                print(f"Skipping optimizer update at epoch {epoch} due to NaNs.")

            arb_loss_epoch += torch.sum(arb_l)
            # loss_epoch += loss
            std_alp_epoch += std_alp
            std_bet_epoch += std_bet
            std_gam_epoch += std_gam
            std_A_epoch += std_A
            std_B_epoch += std_B

        if lr_sched is not None:
            lr_sched.step()

        # evaluate validation efter epoch is run 
        N = len(val_data)    
        G_output, sigma_1, sigma_2, sigma3, rho12, rho13, rho23, mu, r, encoded_mat, encoded = model(val_data)
        reconstructed, p, arb_l, alpha, beta, gamma, A, B, std_alp, std_bet, std_gam, std_A, std_B  = fct(G_output, sigma_1, sigma_2, sigma3, rho12, rho13, rho23, mu, r, encoded_mat, N, model)
        s_final = reconstructed[:, swap_mats0]
        loss_val = criterion(s_final, val_data)

        #
        N = len(train_data)    
        G_output, sigma_1, sigma_2, sigma3, rho12, rho13, rho23, mu, r, encoded_mat, encoded = model(train_data)
        reconstructed, p, arb_l, alpha, beta, gamma, A, B, std_alp, std_bet, std_gam, std_A, std_B  = fct(G_output, sigma_1, sigma_2, sigma3, rho12, rho13, rho23, mu, r, encoded_mat, N, model)
        s_final = reconstructed[:, swap_mats0]
        loss_train = criterion(s_final, train_data)

        # append relevant data
        vallosses_list.append(loss_val.item())
        arb_losses_list.append(arb_loss_epoch.item())
        losses_list.append(loss_train.item())
        current_lr = optimizer.param_groups[0]["lr"] 
    
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], LR:{current_lr:.8f}, Loss: {loss_train.item():.8f}, Arb:{arb_loss_epoch} ")

    std_alp_epoch = std_alp_epoch/(len(train_data)*num_epochs)
    std_bet_epoch = std_bet_epoch/(len(train_data)*num_epochs)
    std_gam_epoch = std_gam_epoch/(len(train_data)*num_epochs)
    std_A_epoch = std_A_epoch/(len(train_data)*num_epochs)
    std_B_epoch = std_B_epoch/(len(train_data)*num_epochs)
    
    return std_alp_epoch, std_bet_epoch, std_gam_epoch, std_A_epoch, std_B_epoch, arb_losses_list, losses_list, vallosses_list

def plot_all(alpha, beta, gamma, A, B, datapoint = None):

    if datapoint is not None:
        alpha = alpha[datapoint, :].reshape(1,30)
        gamma = gamma[datapoint, :].reshape(1,30)
        beta = beta[datapoint, :].reshape(1,30)
        A = A[datapoint, :].reshape(1,30)
        B = B[datapoint, :].reshape(1,30)
        opacity = 1
    else:
        opacity = 0.05


    plt.figure(figsize=(6, 4))
    for i in range(alpha.size(0)):
        plt.plot(range(1,31), alpha.detach()[i].numpy(), color='blue', alpha=opacity)
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r'Time To Maturity')
    plt.title(r'Row-wise $\alpha$ VS Time To Maturity')
    plt.show()
    plt.figure(figsize=(6, 4))
    for i in range(beta.size(0)):
        plt.plot(range(1,31), beta.detach()[i].numpy(), color='blue', alpha=opacity)
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'Time To Maturity')
    plt.title(r'Row-wise $\beta$ VS Time To Maturity')
    plt.show()
    plt.figure(figsize=(6, 4))
    for i in range(gamma.size(0)):
        plt.plot(range(1,31), gamma.detach()[i].numpy(), color='blue', alpha=opacity)   
    plt.ylabel(r'$\gamma$')
    plt.xlabel(r'Time To Maturity')
    plt.title(r'Row-wise $\gamma$ VS Time To Maturity')
    plt.show()
    plt.figure(figsize=(6, 4))
    for i in range(B.size(0)):
        plt.plot(range(1,31), B.detach()[i].numpy(), color='blue', alpha=opacity)
    plt.ylabel(r'$B$')
    plt.xlabel(r'Time To Maturity')
    plt.title(r'Row-wise $B$ VS Time To Maturity')
    plt.show()
    plt.figure(figsize=(6, 4))
    for i in range(A.size(0)):
        plt.plot(range(1,31), A.detach()[i].numpy(), color='blue', alpha=opacity)
    plt.ylabel(r'$A$')
    plt.xlabel(r'Time To Maturity')
    plt.title(r'Row-wise $A$ VS Time To Maturity')
    plt.show()
    plt.figure(figsize=(6, 4))
    for i in range(B.size(0)):
        plt.plot(range(1,31), (gamma.detach()[i].numpy()*(B.detach()[i].numpy())**2), color='blue', alpha=opacity)
    plt.ylabel(r'$\gamma B^2$')
    plt.xlabel(r'Time To Maturity')
    plt.title(r'Row-wise $\gamma B^2$ VS Time To Maturity')
    plt.show()


def plot_constant(A, B, dA, dB, alpha, beta, gamma, datapoint = 0):
    B_np = -B.detach()[datapoint].numpy()
    A_np = -A.detach()[datapoint].numpy()
    dB_np = dB.detach()[datapoint].numpy()
    dA_np = dA.detach()[datapoint].numpy()
    alpha_np = alpha.detach()[datapoint].numpy()
    beta_np = beta.detach()[datapoint].numpy()
    gamma_np = gamma.detach()[datapoint].numpy()

    plt.plot(B_np, dB_np, color='blue', alpha=1)
    for i in range(1, len(B_np)):
        y1 = -alpha_np[i]*0 + beta_np[i]
        y2 = -alpha_np[i]*B_np[i] + beta_np[i]

        x1 = 0
        x2 = B_np[i]

        plt.plot([x1, x2],
                 [y1, y2], color='red', alpha=1)
    plt.xlabel(r'$-B(u)$')
    plt.ylabel(r'$\partial_u B(u)$')    
    plt.show()

    plt.plot(A_np, dA_np, color='blue', alpha=1)
    for i in range(1, len(B_np)):
        g = gamma_np[i]
        
        y1 = g*B_np[i]**2
        y2 = g*B_np[i]**2

        x1 = 0
        x2 = A_np[i]
    
        plt.plot([x1, x2],
                 [y1, y2], color='red', alpha=1)
        
    plt.xlabel(r'$-A(u)$')
    plt.ylabel(r'$\partial_u A(u)$')
    plt.show()

def plot_stepforward(A, B, dA, dB, alpha, beta, gamma, datapoint = 0):
    B_np = -B.detach()[0].numpy()
    A_np = -A.detach()[0].numpy()
    dB_np = dB.detach()[0].numpy()
    dA_np = dA.detach()[0].numpy()
    alpha_np = alpha.detach()[0].numpy()
    beta_np = beta.detach()[0].numpy()
    gamma_np = gamma.detach()[0].numpy()

    plt.plot(B_np, dB_np, color='blue', alpha=1)
    for i in range(1, len(B_np)):
        if i == 0:
            y1 = -alpha_np[i]*B_np[i-1] + beta_np[i]
            y2 = -alpha_np[i]*B_np[i] + beta_np[i]
            x1 = 0
            x2 = B_np[i]
        else:
            y1 = -(alpha_np[i-1]+alpha_np[i])/2*B_np[i-1] + (beta_np[i-1]+beta_np[i])/2
            y2 = -(alpha_np[i-1]+alpha_np[i])/2*B_np[i] + (beta_np[i-1]+beta_np[i])/2
            x1 = B_np[i-1]
            x2 = B_np[i]
        plt.plot([x1, x2],
                 [y1, y2], color='red', alpha=1)
    plt.xlabel(r'$-B(u)$')
    plt.ylabel(r'$\partial_u B(u)$')    
    plt.show()

    plt.plot(A_np, dA_np, color='blue', alpha=1)
    for i in range(1, len(B_np)):
        if i == 0:
            g = gamma_np[i]
        else:
            g = (gamma_np[i-1]+gamma_np[i])/2
        y1 = g*B_np[i]**2
        y2 = g*B_np[i]**2
        x1 = A_np[i-1]
        x2 = A_np[i]
    
        plt.plot([x1, x2],
                 [y1, y2], color='red', alpha=1)
        
    plt.xlabel(r'$-A(u)$')
    plt.ylabel(r'$\partial_u A(u)$')
    plt.show()

def plot_hybrid(A, B, dA, dB, alpha, beta, gamma, datapoint = 0):
    B_np = -B.detach()[0].numpy()
    A_np = -A.detach()[0].numpy()
    dB_np = dB.detach()[0].numpy()
    dA_np = dA.detach()[0].numpy()
    alpha_np = alpha.detach()[0].numpy()
    beta_np = beta.detach()[0].numpy()
    gamma_np = gamma.detach()[0].numpy()

    plt.plot(B_np, dB_np, color='blue', alpha=1)
    for i in range(1, len(B_np)):
        y1 = -alpha_np[i]*0 + beta_np[i]
        y2 = -alpha_np[i]*B_np[i] + beta_np[i]

        x1 = 0
        x2 = B_np[i]

        plt.plot([x1, x2],
                 [y1, y2], color='red', alpha=1)
    plt.xlabel(r'$-B(u)$')
    plt.ylabel(r'$\partial_u B(u)$')    
    plt.show()

    plt.plot(A_np, dA_np, color='blue', alpha=1)
    for i in range(1, len(B_np)):
        if i == 0:
            g = gamma_np[i]
        else:
            g = (gamma_np[i-1]+gamma_np[i])/2
        y1 = g*B_np[i]**2
        y2 = g*B_np[i]**2
        x1 = A_np[i-1]
        x2 = A_np[i]
    
        plt.plot([x1, x2],
                 [y1, y2], color='red', alpha=1)
        
    plt.xlabel(r'$-A(u)$')
    plt.ylabel(r'$\partial_u A(u)$')
    plt.show()

def inverse(row):
    if (row[3] > row[6]):
        return 1
    else: 
        return 0

def plot_latent_inverse_2d(data, encoded):

    X0 = data.copy()


    X0["inversedummy"] = data.apply(inverse, axis=1)

    label_encoder = LabelEncoder()
    color_encoded = label_encoder.fit_transform(X0['inversedummy'])  
    palette = sns.color_palette("tab10", n_colors=len(label_encoder.classes_))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    colors = [palette[i] for i in color_encoded]

    sc = ax.scatter(encoded.numpy()[:, 0], encoded.numpy()[:, 1],
                alpha=0.7, c=colors)  

    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_title(r"$z_t$ latent space representation")


    legend_patches = [mpatches.Patch(color=palette[i], label=label)
                  for i, label in enumerate(label_encoder.classes_)]

    plt.legend(handles=legend_patches, title="Is Inverted", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def plot_latent_inverse_3d(data, encoded, x, y, z):

    X0 = data.copy()   

    X0["inversedummy"] = data.apply(inverse, axis=1)

    label_encoder = LabelEncoder()
    color_encoded = label_encoder.fit_transform(X0['inversedummy'])  
    palette = sns.color_palette("tab10", n_colors=len(label_encoder.classes_))

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(111, projection='3d')

    colors = [palette[i] for i in color_encoded]

    sc = ax.scatter(encoded.numpy()[:, x], encoded.numpy()[:, y], encoded.numpy()[:, z],
                alpha=0.7, c=colors)  

    ax.set_xlabel(fr"$z_{{{x+1}}}$")
    ax.set_ylabel(fr"$z_{{{y+1}}}$")
    ax.set_zlabel(fr"$z_{{{z+1}}}$")
    ax.set_title(r"$z_t$ latent space representation")

    legend_patches = [mpatches.Patch(color=palette[i], label=label)
                  for i, label in enumerate(label_encoder.classes_)]

    plt.legend(handles=legend_patches, title="Is Inverted", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def dummy_inverse(data):
    X0 = data.copy()    

    X0["inversedummy"] = data.apply(inverse, axis=1)

    return X0
