import gym
from gym import wrappers
import numpy as np
import random, tempfile, os
from collections import deque, namedtuple
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
import tensorflow.keras.backend as K
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from scipy.stats import rankdata

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'error'])

class SumTree:
    """
    Binary sum tree for PER (Prioritized Experience Replay)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def total(self):
        return self.tree[0]

class PrioritizedMemory:
    """
    Prioritized Experience Replay memory
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.max_priority = 1.0

    def _get_priority(self, error):
        return (np.abs(error) + 1e-6) ** self.alpha

    def add(self, experience):
        priority = self.max_priority
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        return batch, idxs, is_weight

    def update(self, idx, error):
        priority = self._get_priority(error)
        self.max_priority = max(self.max_priority, priority)
        self.tree.update(idx, priority)

class DuelingDQN:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        input_layer = Input(shape=(self.state_dim,))
        
        # Shared layers
        shared = Dense(512, activation='relu')(input_layer)
        shared = Dense(256, activation='relu')(shared)

        # Value stream
        value = Dense(128, activation='relu')(shared)
        value = Dense(1)(value)

        # Advantage stream
        advantage = Dense(128, activation='relu')(shared)
        advantage = Dense(self.action_dim)(advantage)

        # Combine streams
        output = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)),
                       output_shape=(self.action_dim,))([value, advantage])

        model = tf.keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, states, targets, importance_weights):
        self.model.fit(states, targets, sample_weight=importance_weights, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def target_predict(self, state):
        return self.target_model.predict(state)

class NoisyDense(Dense):
    """
    Noisy Dense Layer for exploration
    """
    def __init__(self, units, sigma_init=0.017, **kwargs):
        self.sigma_init = sigma_init
        super(NoisyDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        self.kernel_sigma = self.add_weight(name='kernel_sigma',
                                          shape=(input_shape[1], self.units),
                                          initializer=tf.keras.initializers.Constant(self.sigma_init),
                                          trainable=True)
        
        self.bias_sigma = self.add_weight(name='bias_sigma',
                                        shape=(self.units,),
                                        initializer=tf.keras.initializers.Constant(self.sigma_init),
                                        trainable=True)

        super(NoisyDense, self).build(input_shape)

    def call(self, inputs):
        kernel_noise = tf.random.normal(self.kernel.shape)
        bias_noise = tf.random.normal(self.bias.shape)
        
        kernel = self.kernel + self.kernel_sigma * kernel_noise
        bias = self.bias + self.bias_sigma * bias_noise
        
        return K.dot(inputs, kernel) + bias

class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim, 
                 learning_rate=0.001,
                 gamma=0.99,
                 memory_size=100000,
                 batch_size=64,
                 target_update_freq=1000,
                 n_step=3):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        
        # Initialize Dueling DQN
        self.model = DuelingDQN(state_dim, action_dim, learning_rate)
        
        # Initialize Prioritized Experience Replay
        self.memory = PrioritizedMemory(memory_size)
        
        # N-step transition buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.steps = 0
        
        # Metrics
        self.loss_history = []
        self.reward_history = []

    def _preprocess_state(self, state):
        # Normalize state
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def get_action(self, state):
        state = self._preprocess_state(state)
        q_values = self.model.predict(state.reshape(1, -1))[0]
        return np.argmax(q_values)

    def _get_n_step_info(self, n_step_buffer):
        state = n_step_buffer[0].state
        action = n_step_buffer[0].action
        reward = sum([self.gamma**i * t.reward for i, t in enumerate(n_step_buffer)])
        next_state = n_step_buffer[-1].next_state
        done = n_step_buffer[-1].done
        return Experience(state, action, reward, next_state, done, 0)

    def remember(self, state, action, reward, next_state, done):
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        
        experience = Experience(state, action, reward, next_state, done, 0)
        self.n_step_buffer.append(experience)
        
        if len(self.n_step_buffer) == self.n_step:
            n_step_experience = self._get_n_step_info(self.n_step_buffer)
            self.memory.add(n_step_experience)

    def train(self):
        if len(self.memory.tree.data) < self.batch_size:
            return
        
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        # Double DQN update
        current_q_values = self.model.predict(next_states)
        next_actions = np.argmax(current_q_values, axis=1)
        next_q_values = self.model.target_predict(next_states)
        
        targets = self.model.predict(states)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma**self.n_step * \
                                       next_q_values[i][next_actions[i]]
            
            # Update priorities
            error = abs(targets[i][actions[i]] - current_q_values[i][actions[i]])
            self.memory.update(idxs[i], error)
        
        # Train the model
        loss = self.model.train(states, targets, is_weights)
        self.loss_history.append(loss)
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.model.update_target_model()

    def save(self, filepath):
        self.model.model.save(filepath)

    def load(self, filepath):
        self.model.model = tf.keras.models.load_model(filepath)
        self.model.update_target_model()

class ParallelEnv:
    def __init__(self, env_name, num_envs=4):
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs
        
    def reset(self):
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            states = list(executor.map(lambda env: env.reset(), self.envs))
        return np.array(states)
    
    def step(self, actions):
        results = []
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            for env_idx, (env, action) in enumerate(zip(self.envs, actions)):
                future = executor.submit(env.step, action)
                results.append(future)
                
        states, rewards, dones, infos = zip(*[r.result() for r in results])
        return np.array(states), np.array(rewards), np.array(dones), infos

def train(env_name="LunarLander-v2", episodes=1000, render=False):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = EnhancedDQNAgent(state_dim, action_dim)
    parallel_env = ParallelEnv(env_name)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            
        agent.reward_history.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
            
    env.close()
    return agent

if __name__ == "__main__":
    agent = train(episodes=1000, render=True)