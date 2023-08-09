import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.initializers import GlorotNormal
from tensorflow.keras import regularizers

# CSV and TXT file paths here
act_path = ""
env_path = ""
atom_path = ""

actions = pd.read_csv(act_path)
env = pd.read_csv(env_path)
yo = np.loadtxt(atom_path)

def softplus(x):
    return np.log(np.exp(x) + 1)

X_env = env.values  # Environmental parameters
actions = actions.values  # Control parameters (actions)

# Scaling. Can expiriment with other scalings as well
scaler = StandardScaler()
scaler.fit(X_env)
X = scaler.transform(X_env)  # Scaling the environmental parameters
y = yo.reshape(-1, 1)
yscaler = StandardScaler()
yscaler.fit(y)
y = yscaler.transform(y)

ascaler = StandardScaler()
ascaler.fit(actions)
actions = ascaler.transform(actions)

input_shape = X_env.shape[1]
action_space_size = 30
dro = 0.01  # Dropout (optional, likely not needed)


# initialize the model , can try different archetectures if you want!
inputs = tf.keras.Input(shape=input_shape+3)
hidden_layer_1 = tf.keras.layers.Dense(64, activation='selu', kernel_initializer=GlorotNormal(), kernel_regularizer=regularizers.L1(0.0175))(inputs)
dropout_1 = tf.keras.layers.Dropout(dro)(hidden_layer_1)  
hidden_layer_2 = tf.keras.layers.Dense(64, activation='selu', kernel_initializer=GlorotNormal(), kernel_regularizer=regularizers.L1(0.0175))(dropout_1)
dropout_2 = tf.keras.layers.Dropout(dro)(hidden_layer_2) 
hidden_layer_3 = tf.keras.layers.Dense(64  , activation='selu', kernel_initializer=GlorotNormal(), kernel_regularizer=regularizers.L1(0.0175))(dropout_2)
dropout_3 = tf.keras.layers.Dropout(dro)(hidden_layer_3)  
hidden_layer_4 = tf.keras.layers.Dense(64  , activation='selu', kernel_initializer=GlorotNormal(), kernel_regularizer=regularizers.L1(0.0175))(dropout_3)
dropout_4 = tf.keras.layers.Dropout(dro)(hidden_layer_4)  

outs = tf.keras.layers.Dense(1+action_space_size*2, activation="linear", name='outs')(dropout_4) # critic and actor together
model = tf.keras.Model(inputs=inputs, outputs=[outs])

# dummy network for the very first point's partial observabillity
dummy_inputs = tf.keras.Input(shape=input_shape)
hidden_layer_1 = tf.keras.layers.Dense(64, activation='selu', kernel_initializer=GlorotNormal())(dummy_inputs)
dropout_1 = tf.keras.layers.Dropout(dro)(hidden_layer_1)  
hidden_layer_2 = tf.keras.layers.Dense(64, activation='selu', kernel_initializer=GlorotNormal())(dropout_1)
dropout_2 = tf.keras.layers.Dropout(dro)(hidden_layer_2) 
hidden_layer_3 = tf.keras.layers.Dense(64  , activation='selu', kernel_initializer=GlorotNormal())(dropout_2)
dropout_3 = tf.keras.layers.Dropout(dro)(hidden_layer_3)    
hidden_layer_4 = tf.keras.layers.Dense(64  , activation='selu', kernel_initializer=GlorotNormal())(dropout_3)
dropout_4 = tf.keras.layers.Dropout(dro)(hidden_layer_4)  

outs = tf.keras.layers.Dense(1+action_space_size*2, activation="linear", name='outs')(dropout_4) # critic and actor together
model_dummy = tf.keras.Model(inputs=dummy_inputs, outputs=[outs])


opt = keras.optimizers.Adam(0.00001 ) # low initial leaning rate helps keep it stable in the face of exploding gradients

# param bounds for our system. Can use for Tanh or hard walls
lbO = np.array([0.34,10,10,5,8,-5,-5,-5,-5,-9.5,-5,-7,103.95,103.90,0.1,0.1,0.1,0.1,-9,-9,-9,-9,-9,-9,-9,-9,-9,0.1,105.0,104.2]) #
rbO = np.array([0.90,22,22,22,22,5,5,5,5,9.5,5,9,104.3,104.29,1,1,1,1,9,9,9,9,9,9,9,9,9,1,107.5,106])

lb = ascaler.transform(np.reshape(lbO,(1,-1))).flatten()
rb = ascaler.transform(np.reshape(rbO,(1,-1))).flatten()


def new_loss(y_true, y_pred): # this is the all important loss function
    
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    value_true, action_true, extra = y_true[:,0], y_true[:,1:action_space_size+1], y_true[:,action_space_size+1:]
    value_pred, mu_pred, sigma_pred = y_pred[:,0], y_pred[:,1:action_space_size+1], y_pred[:,action_space_size+1:]
    ko = 0.001 # weigthing of actor and critic. can play around with (will potentially explode if made > 0.01 ish)
    
    # keep it in bounds. Could use tanh instead too as mentioned in paper. Not much difference found either way
    mu_pred = tf.maximum(mu_pred,lb ) 
    mu_pred = tf.minimum(mu_pred,rb )
    sigma_pred = tf.math.softplus(sigma_pred) # keeping variance positive
    eps = 1e-17 # failsafe keeps gradients from exploding
    sigma_pred = tf.math.add(sigma_pred,eps)  # keeping variance above machine precision limit
    dist = tf.compat.v1.distributions.Normal(loc=mu_pred, scale=sigma_pred) # policy
    log_prob = tf.math.log(tf.maximum(dist.prob(action_true), eps)) # log probabillity
    TD_target = value_true # discount factor = 0
    TD_error = tf.math.subtract(TD_target, value_pred) # difference between expected and actual for whole batch
    
    critic_loss = tf.math.square(TD_error) # simply a squared error
    actor_loss = -tf.reduce_sum(log_prob, axis=1) * TD_error # actor loss 

    k = ko        
    loss = critic_loss + k*actor_loss # unequal weighting
    loss = tf.reduce_mean(loss) # taking mean of all batches
    
    return loss


def dummy_loss(y_true, y_pred): # for the dummy network, only important for very first point
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    value_true, action_true, extra = y_true[:,0], y_true[:,1:action_space_size+1], y_true[:,action_space_size+1:]
    value_pred, mu_pred, sigma_pred = y_pred[:,0], y_pred[:,1:action_space_size+1], y_pred[:,action_space_size+1:]
    ko = 0.01
    
    mu_pred = tf.maximum(mu_pred,lb )
    mu_pred = tf.minimum(mu_pred,rb )
    sigma_pred = tf.math.softplus(sigma_pred)
    eps = 1e-12
    sigma_pred = tf.math.add(sigma_pred,eps)
    dist = tf.compat.v1.distributions.Normal(loc=mu_pred, scale=sigma_pred)
    log_prob = tf.math.log(tf.maximum(dist.prob(action_true), eps))
        
    TD_target = value_true # discount factor = 0
    TD_error = tf.math.subtract(TD_target, value_pred)
    critic_loss = tf.math.square(TD_error) 
    actor_loss = -tf.reduce_sum(log_prob, axis=1) * TD_error    
    k = ko
    loss = critic_loss + k*actor_loss # equal weighting for now
    loss = tf.reduce_mean(loss)

    return loss
    
model.compile(opt,new_loss)
model_dummy.compile(opt,dummy_loss)


batch_size= 64

extras = np.zeros(actions.shape)
next_pred = None
losses = []
nextsug = [] 


for i in range(1, len(X)-batch_size*1): # the "faked" online training loop
# often takes a while to run on full set    
# this loop includes the extra fancy partial observabillity compensating stuff
# does not need to be this complicated if want basic vanilla actor-critic

    prev = X[i-1:i+batch_size-1]
    batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    a_batch = actions[i:i+batch_size]
    last_actions = actions[i-1:i+batch_size-1]
    last_y = y[i-1:i+batch_size-1]

    if i == 1: # since we do partial observability, need this to make a faked point for the first batch
        pred = model_dummy.predict(prev,batch_size, verbose = 0)
        value_pred, mu_pred, sigma_pred = pred[:,0].reshape(-1,1), pred[:,1:action_space_size+1], pred[:,action_space_size+1:]
        sigma_pred = softplus(sigma_pred)
        
        prob = (last_actions - mu_pred)**2 / (2 * sigma_pred**2)
        prob = np.sum(prob, axis = -1)
        all_sig = np.product(sigma_pred,axis = -1)
        prob = 1/((all_sig) * np.sqrt((2 * np.pi)**(mu_pred.shape[-1]))) * np.exp( -prob )    
        sigsig = np.std(prob, axis = 0)
        mumu = np.mean(prob, axis = 0)
    
    else: # normal partial observabiliity procedure
        value_pred, mu_pred, sigma_pred = next_pred[:,0].reshape(-1,1), next_pred[:,1:action_space_size+1], next_pred[:,action_space_size+1:]
        sigma_pred = softplus(sigma_pred)
        prob = (last_actions - mu_pred)**2 / (2 * sigma_pred**2) # normal dist
        prob = np.sum(prob, axis = -1) # sum over policy dimension to get 1 value
        all_sig = np.product(sigma_pred,axis = -1) # product over policy dimension to get 1 value
        prob = np.maximum(1/((all_sig) * np.sqrt((2 * np.pi)**(mu_pred.shape[-1]))) * np.exp( -prob ), 1e-30) # normalizing normal dist (also not letting it get too small below machine precision)
        if np.any(np.isnan(prob)): # this should never trigger
            prob = np.ones(len(prob)) * 1e-30
            
    prob =  ((prob-mumu)/sigsig).reshape(-1,1) # pre process previous actor prob of choosing action
    V = value_pred.reshape(-1,1) # previous critic prediciton
    Xin = np.concatenate((batch,prob*0,V, last_y), axis = -1) # adding all the extra stuff to environment. Can play around with what to include or not
    out = np.concatenate((y_batch,a_batch,extras[i:i+batch_size]), axis = 1)
    loss = model.train_on_batch(Xin, out) # update network
    losses.append(loss)
    next_pred = model.predict(Xin,verbose = 0) # next prediction
    nextsug.append(next_pred[-1]) # just for plotting of policy purposes
     
        

#%%

# plots a dimension of the policy, and critic prediction

b = 64*4 # the last b points
col = 12 # which dimension of policy you want to see
q = np.array(nextsug[-b:])[:,1:action_space_size+1]
s = ascaler.inverse_transform(q) 

qs = np.array(nextsug[-b:])[:,action_space_size+1:]
qs = softplus(qs)
qp = q + qs
qm = q - qs
try:
    qp = ascaler.inverse_transform(qp)
    qm = ascaler.inverse_transform(qm)
except:
    qm = np.zeros(q.shape)
    qp = np.zeros(q.shape)
    
    
std = (qp-qm)/2
x = np.arange(0,b,1)

plt.figure()
plt.plot(x,s[:,col], marker = ".", label = "policy mean", color = (0.0, 0.3843137254901961, 0.19215686274509805) )
plt.fill_between(x[:],s[:,col]-std[:,col], s[:,col]+std[:,col], facecolor= (0.9019607843137255,0.788235294117647,0.38823529411764707), alpha=0.8, edgecolor='white',linewidth=1, linestyle='dashed', label = r"policy uncertinaty (1 $\sigma$)")
plt.ylabel("Agent's policy - MOT cooling frequency (MHz)")
plt.xlabel("Run number")
plt.tight_layout()
plt.legend(loc = "upper left", fancybox = True, frameon=True, framealpha = 0.95)
ax = plt.gca()
ax.tick_params(axis = 'both',width =1, length = 3,direction = 'in',top='true',right = 'true')


plt.figure()
V = np.array(nextsug[-b:])[:,0]
V = yscaler.inverse_transform(V.reshape(-1,1))
plt.plot(V, label = "critic pred")
plt.xlabel("Run number")
plt.ylabel("log(N)")
plt.legend()
