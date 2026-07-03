import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import shap
import copy
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


# ## Mortality to Resistance


pm = pd.read_csv("malaria-gan-outputs/random_pm.csv")
pm_values = pm.values.flatten().tolist()
percentiles = [0, 5, 10, 15, 50, 85, 90, 95, 100]
pm_percentile_values = np.percentile(pm_values, percentiles)
percentile_results = dict(zip([f"{p}%" for p in percentiles], [round(pmpv,1) for pmpv in pm_percentile_values]))
percentile_results


# 25th and 75th percentiles for both sigmoid and data
# 75% of data located between 60 and 66
def pm_to_R(pm):
    pm = pm/100
    return 1/(1+np.exp(36*(pm-0.63)))

for ppm in percentile_results.values():
    print(round(ppm,0),round(pm_to_R(ppm),3))


ts = []
for row in range(10000):
    data = pm.loc[row].tolist()
    interpol = []
    for i in range(7):
        start = data[i]
        end = data[i+1]
        seg = np.linspace(start, end, 26, endpoint=False)
        interpol.extend(seg)
    interpol.append(data[-1])
    interpol = interpol[:180]
    ts.append([pm_to_R(ip) for ip in interpol])
ts = pd.DataFrame(ts)


samp_ts = ts.sample(n=250,random_state=42)
plt.figure(figsize=(6,4.5))
for i in range(len(samp_ts)):
    plt.plot(samp_ts.columns, samp_ts.iloc[i], alpha=0.75, linewidth=0.3)
plt.xlabel("Day")
plt.ylabel("Resistance")
plt.title("Synthetic Resistance Impact")
plt.show()


# ## County Setup


class MalariaCounty(gym.Env):
    def __init__(self,pop_multiply=1,vegetation_biting=0.3,init_coverage=0.4,treatment_seeking=10):
        super(MalariaCounty,self).__init__()
        self.t = 0
        self.tau = 0
        self.Rs = ts.loc[random.randint(0,9999)]
        self.multiplier = pop_multiply
        ##### Constants #####
        self.mu_H = 1/(60*365)  # 60 year life expectancy
        self.mu_M = 1/12          # 12 days lifespan
        self.gamma = 1/50         # immunity waning
        self.gamma_T = 1/21       # 3 weeks
        self.gamma_eta = 1/365    # 1 year
        self.rho = 1/treatment_seeking  # 10 days avg
        self.delta_H = 0.01               # human
        self.delta_T = 0.001              # treated
        self.sigma_H = 1/14             # incubation humans
        self.sigma_M = 1/10             # incubation mosquitoes
        self.epsilon = 0.3                # treated humans less infectious
        self.a = vegetation_biting                                # biting rate per mosquito per day
        self.a_t = self.a*(1+0.2*np.sin(2*np.pi*self.t/180))  # 20% seasonal amplitude every 180 days
        self.b = 0.2                                              # transmission mosquito -> human
        self.c = 0.2                                              # transmission human -> mosquito
        self.external_seeding = 60*self.multiplier                # 80 new exposed people per day
        self.C = init_coverage             # ITN initial coverage
        self.delta = 0.001                 # ITN decay rate
        self.R = self.Rs[self.t]           # resistance impact
        self.N_H = 10000*self.multiplier   # total human population
        self.N_M = 30000*self.multiplier   # mosquito population is 3x human population
        # Initial values and action space
        self.reset()
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)

    def step(self, action):
        # Daily updates
        curr_itns = self.C*self.N_H
        new_itns = float(np.clip(action[0],0,self.N_H))
        total_itns = curr_itns + new_itns
        self.C = min(float(total_itns/self.N_H),1)
        self.tau = self.tau*curr_itns/total_itns
        self.R = self.Rs[self.t]
        self.a_t = self.a*(1+0.2*np.sin(2*np.pi*self.t/180))
        # ITN impact on infection from mosquitoes to humans
        theta_ITN = self.C*np.exp(-self.delta*self.tau)*(1-0.5*self.R)
        # Updated force of infection
        Lambda_H = self.a_t * self.c * self.I_M/self.N_M * max(1-theta_ITN,0)
        Lambda_M = self.a_t * self.b * (self.I_H+self.epsilon*self.T_H)/self.N_H
        # Human differential equations
        dS_H = self.mu_H*self.N_H - Lambda_H*self.S_H + self.gamma*self.R_H - self.mu_H*self.S_H
        dE_H = Lambda_H*self.S_H - self.sigma_H*self.E_H - self.mu_H*self.E_H
        dE_H += self.external_seeding
        dI_H = self.sigma_H*self.E_H - (self.rho+self.mu_H + self.delta_H)*self.I_H
        dT_H = self.rho*self.I_H - (self.mu_H+self.delta_T + self.gamma_T)*self.T_H
        dR_H = self.gamma_T*self.T_H - self.gamma*self.R_H - self.mu_H*self.R_H
        # Mosquito differential equations
        dS_M = self.mu_M*self.N_M - Lambda_M*self.S_M - self.mu_M*self.S_M
        dE_M = Lambda_M*self.S_M - self.sigma_M*self.E_M - self.mu_M*self.E_M
        dI_M = self.sigma_M*self.E_M - self.mu_M*self.I_M
        # Euler integration step
        dt = 1
        self.S_H += dS_H*dt
        self.E_H += dE_H*dt
        self.I_H += dI_H*dt
        self.T_H += dT_H*dt
        self.R_H += dR_H*dt
        self.S_M += dS_M*dt
        self.E_M += dE_M*dt
        self.I_M += dI_M*dt
        self.t += dt
        self.tau += dt
        # Observation space
        obs = np.array([self.S_H, self.E_H, self.I_H, self.T_H, self.R_H,
                        self.S_M, self.E_M, self.I_M, self.a_t, self.R], dtype=np.float32)
        # Reward function
        reward = -self.E_H-self.I_H  # minimize number of infectious humans
        done = False
        return obs,reward,done,{}

    def reset(self):
        # Humans
        self.S_H = (self.N_H-1400)*self.multiplier   # Majority uninfected
        self.E_H = 1000*self.multiplier              # Incubating but not yet infectious
        self.I_H = 200*self.multiplier               # Actively transmitting
        self.T_H = 100*self.multiplier               # Undergoing treatment
        self.R_H = 100*self.multiplier               # Temporary immunity
        # Mosquitoes
        self.S_M = (self.N_M-6000)*self.multiplier   # Susceptible mosquitoes
        self.E_M = 3000*self.multiplier              # Infected, not yet infectious
        self.I_M = 3000*self.multiplier              # Capable of transmitting
        self.t = 0 # Time step (day)
        self.itn_t = 0 # ITN Time


def visualize_county(history,titles='Malaria Dynamics'):
    # All compartments
    labels = ["Susceptible Humans","Exposed Humans","Infectious Humans","Treated Humans","Recovered Humans","Susceptible Mosquitoes","Exposed Mosquitoes","Infectious Mosquitoes"]
    plt.figure(figsize=(4,2.8))
    line_handles = []
    for i in range(8):
        line, = plt.plot(history[:, i], label=labels[i])
        line_handles.append(line)
    colors = [line.get_color() for line in line_handles]
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title(titles)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.025))
    plt.grid(True)
    plt.show()
    # Non-susceptible compartments
    plot_indices = [1, 2, 3, 4, 6, 7]
    labels = ["Exposed Humans","Infectious Humans","Treated Humans","Recovered Humans","Exposed Mosquitoes","Infectious Mosquitoes"]
    plt.figure(figsize=(4,2.8))
    for idx in plot_indices:
        plt.plot(history[:, idx], label=labels[plot_indices.index(idx)], color=colors[idx])
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title(titles)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1.025))
    plt.grid(True)
    plt.show()


# Simulation
control_county = MalariaCounty()
control_county.reset()
num_days = 180
history = []
# Main loop
for day in range(num_days):
    if day not in [30,80,130]:
        action = np.array([0],dtype=np.float32)  # Fixed ITN coverage
    else:
        action = np.array([0],dtype=np.float32) # -315k vs -295k with 2500
    obs,reward,done,_ = control_county.step(action)
    history.append(obs.copy())
# Results
full_history = np.array(history)
history = full_history[:,:-2]


visualize_county(history,titles="Baseline Malaria Dynamics")


# ## County Allocation Setup


class CountyAllocation(gym.Env):
    def __init__(self):
        super(CountyAllocation, self).__init__()
        self.counties = [
            MalariaCounty(pop_multiply=1.5, vegetation_biting=0.1, init_coverage=0.5, treatment_seeking=5),  # Urban
            MalariaCounty(pop_multiply=1.0, vegetation_biting=0.3, init_coverage=0.4, treatment_seeking=10), # Mixed
            MalariaCounty(pop_multiply=0.7, vegetation_biting=0.5, init_coverage=0.3, treatment_seeking=15), # Rural
            MalariaCounty(pop_multiply=1.2, vegetation_biting=0.2, init_coverage=0.35, treatment_seeking=7), # Semi-urban
            MalariaCounty(pop_multiply=0.8, vegetation_biting=0.4, init_coverage=0.45, treatment_seeking=12),# Rural
        ]
        low_obs = np.array([0.0]*((4 + 11)*5), dtype=np.float32)
        high_obs = np.array([np.inf]*((4 + 11)*5), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)

    def reset(self):
        obs = []
        for county in self.counties:
            county.reset()
            pop_mult = county.multiplier
            veg_bite = county.a
            init_cov = county.C
            treat_seek = 1 / county.rho
            state = [county.S_H, county.E_H, county.I_H, county.T_H, county.R_H,
                     county.S_M, county.E_M, county.I_M, county.a_t, county.R]
            obs.extend([pop_mult, veg_bite, init_cov, treat_seek] + state)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        obs = []
        total_infections = 0.0
        for i, county in enumerate(self.counties):
            county_obs, _, _, _ = county.step(np.array([action[i]], dtype=np.float32))
            pop_mult = county.multiplier
            veg_bite = county.a
            init_cov = county.C
            treat_seek = 1 / county.rho
            state = [county.S_H, county.E_H, county.I_H, county.T_H, county.R_H,
                     county.S_M, county.E_M, county.I_M, county.a_t, county.R]
            obs.extend([pop_mult,veg_bite,init_cov,treat_seek] + state)
            total_infections += county.E_H
            total_infections += county.I_H
        reward = -total_infections  # minimize infectious humans
        done = False
        return np.array(obs, dtype=np.float32), reward, done, {}


baseline = {
    30: np.array([0,0,0,0,0], dtype=np.float32),
    80: np.array([0,0,0,0,0], dtype=np.float32),
    130: np.array([0,0,0,0,0], dtype=np.float32),
}
# 5.175
population_static1 = {
    30: np.array([3750, 2500, 1750, 3000, 2000], dtype=np.float32),
    80: np.array([3750, 2500, 1750, 3000, 2000], dtype=np.float32),
    130: np.array([3750, 2500, 1750, 3000, 2000], dtype=np.float32),
}
# 4.886
population_static2 = {
    30: np.array([7500, 5000, 3500, 6000, 4000], dtype=np.float32),
    80: np.array([7500, 5000, 3500, 6000, 4000], dtype=np.float32),
    130: np.array([7500, 5000, 3500, 6000, 4000], dtype=np.float32),
}
# 4.650


# Simulation with ITN allocation
practice_env = CountyAllocation()
practice_env.reset()
num_days = 180
history = []
rewards = [0,0,0,0]
reward_ind = 0
last_obs = []
# Main loop
for day in range(num_days):
    if day in [30,80,130]:
        action = population_static1[day]
        reward_ind += 1
    else:
        action = np.zeros(5, dtype=np.float32)
    obs, reward, done, _ = practice_env.step(action)
    last_obs = list(obs)
    rewards[reward_ind] += reward/52000
    history.append(obs.copy())
# Results
full_history = np.array(history)
cnty1_hist = full_history[:,4:12]
cnty2_hist = full_history[:,18:26]
cnty3_hist = full_history[:,32:40]
cnty4_hist = full_history[:,46:54]
cnty5_hist = full_history[:,60:68]
print([r/rewards[0] for r in rewards[1:]])
print(round(sum(r/rewards[0] for r in rewards[1:]),3))


visualize_county(cnty5_hist,titles="Static Allocation Malaria Dynamics")


# ## State-Action Regression


class SAR(nn.Module):
    def __init__(self, input_dim=75, hidden_dim=128, dropout_rate=0.01):
        super(SAR, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_dim, 1)  # scalar output for reward prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.output(x)


# epsilon-greedy policy
def select_action(state, epsilon):
    if np.random.rand()<epsilon:
        probs = np.random.dirichlet(np.ones(5))
        return (probs*5000).astype(np.float32)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    best_action = None
    best_reward = -np.inf
    for _ in range(25):
        candidate = np.random.dirichlet(np.ones(5))*13000
        candidate_tensor = torch.tensor(candidate, dtype=torch.float32).unsqueeze(0)
        combined = torch.cat([state_tensor, candidate_tensor], dim=1)
        with torch.no_grad():
            predicted_reward = sar(combined).item()
        if predicted_reward>best_reward:
            best_reward = predicted_reward
            best_action = candidate

    return best_action.astype(np.float32)


def train_step():
    if len(memory)<batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    state_action_input = torch.cat([states,actions], dim=1)  # shape: [batch_size, 75]
    predicted_rewards = sar(state_action_input)
    loss = loss_fn(predicted_rewards,rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


memory = deque(maxlen=10000)
gamma = 0.99
batch_size = 128
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.996
lr = 0.001
tau = 0.01
sar = SAR()
target_sar = SAR()
target_sar.load_state_dict(sar.state_dict())
optimizer = optim.Adam(sar.parameters(),lr=lr)
loss_fn = nn.MSELoss()
num_episodes = 1000
training_infections = []
best_ratio = 1000
best_allocations = None
best_scale = None


for episode in range(num_episodes):
    allocation_env = CountyAllocation()
    allocation_env.reset()
    infected_hist = [0,0,0,0]
    infected_ind = 0
    last_obs,main_actions,main_obs = [],[],[]

    for day in range(180):
        if day in [30,80,130]:
            action = select_action(last_obs, epsilon)
            main_obs.append(last_obs)
            main_actions.append(action)
            infected_ind += 1
        else:
            action = np.zeros(5, dtype=np.float32)
        new_obs, reward, done, _ = allocation_env.step(action)
        infected_hist[infected_ind] += reward / 52000

        if day in [79,129,179]:
            index = int((day-79)//50)
            c_reward = infected_hist[index+1]/infected_hist[0]  # already negative
            c_action = main_actions[index]
            c_obs = main_obs[index]
            action_probs = c_action / (np.sum(c_action)+1e-8)
            l2_penalty = np.sum((action_probs-0.2)**2)
            adjusted_reward = c_reward - 0.04*l2_penalty
            memory.append((c_obs, c_action, -adjusted_reward))
            train_step()
            if index == 2:
                episode_allocations = list(zip([79,129,179], infected_hist[1:], main_actions))
                scale = infected_hist[0]
                cum_ratio = sum([episode_allocations[i][1] for i in range(3)])/scale
                if cum_ratio < best_ratio:
                    best_scale = scale
                    best_ratio = cum_ratio
                    best_allocations = episode_allocations
        last_obs = new_obs

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if (episode+1)%100==0:
        print(f"Episode {episode+1}, Epsilon: {round(epsilon, 3)}",infected_hist)
    training_infections.append([episode_allocations[i][1]/scale for i in range(3)])


# 4.539
# 4.597
print("===== Best Allocation Found =====")
print(f"Cumulative Infection Ratio: {best_ratio:.3f}")
print(f"Scaling Factor (Day 0 to 29): {best_scale:.3f}")
for day, raw_infected, action in best_allocations:
    formatted_action = np.array2string(action, precision=0, suppress_small=True, separator=' ')
    print(f"Day {day}: Raw = {raw_infected:.3f}, Action = {formatted_action}, Adjusted (scaled) = {raw_infected/best_scale:.3f}")


inf_array = np.array(training_infections)
w_size = 20
n_windows = len(inf_array)//w_size
avg_ratio = inf_array[:n_windows*w_size].reshape(n_windows,w_size, 3).mean(axis=1)
x = np.arange(1,n_windows+1)*w_size

plt.figure(figsize=(5, 3.5))
plt.plot(x, avg_ratio[:,0], label="Day 30", alpha=0.8)
plt.plot(x, avg_ratio[:,1], label="Day 80", alpha=0.8)
plt.plot(x, avg_ratio[:,2], label="Day 130", alpha=0.8)
plt.xlabel("Episode Number")
plt.ylabel("Cumulative Infection Ratio")
plt.title("Training Loss Averaged Every 20 Episodes")
plt.legend(loc='upper right', bbox_to_anchor=(1,1))
plt.grid(True)
plt.show()


vanilla_names = ["S_H", "E_H", "I_H", "T_H", "R_H", "S_M", "E_M", "I_M", "a(t)", "R", "pop_mult", "veg_bite", "init_cov", "treat_seek"]
all_names = [f"C{i} {name}" for i in range(1,6) for name in vanilla_names]
for i in range(1,6):
    all_names.append(f"C{i} ITNs")


# ## SHAP Interpretability


sar_nodrop = copy.deepcopy(sar)
sar_nodrop.eval()
for module in sar_nodrop.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.0
X = np.array([
    np.concatenate([m[0], m[1]]) for m in memory
    if isinstance(m[0],(list, np.ndarray)) and isinstance(m[1],(list, np.ndarray))
])
X_sample = X[np.random.choice(len(X), size=600, replace=False)]
X_tensor = torch.tensor(X_sample, dtype=torch.float32)
explainer = shap.DeepExplainer(sar_nodrop, X_tensor)
shap_values = explainer.shap_values(X_tensor, check_additivity=False)


shap.summary_plot(shap_values, features=X_sample, feature_names=all_names, max_display=11)


# ## Decision Tree Interpretability


X = np.array([
    np.concatenate([m[0], m[1]]) for m in memory
    if isinstance(m[0], (list, np.ndarray)) and isinstance(m[1], (list, np.ndarray))
])
y = sar(torch.tensor(X, dtype=torch.float32)).detach().numpy()
print("SAR prediction range:", y.min(), "to", y.max())
print("Mean:", y.mean(), "Std:", y.std())
vanilla_names = ["S_H", "E_H", "I_H", "T_H", "R_H", "S_M", "E_M", "I_M", "a(t)", "R", "pop_mult", "veg_bite", "init_cov", "treat_seek"]
all_names = [f"C{i} {name}" for i in range(1, 6) for name in vanilla_names]
for i in range(1, 6):
    all_names.append(f"C{i} ITNs")
tree = DecisionTreeRegressor(max_depth=2,random_state=42)
tree.fit(X, y)
tree.score(X,y) # 0.266


plt.figure(figsize=(9,4))
plot_tree(tree, feature_names=all_names, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Surrogate for SAR Policy")
plt.show()
