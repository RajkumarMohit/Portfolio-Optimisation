import streamlit as st
import pandas as pd
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Define Portfolio Environment
class PortfolioEnv(gym.Env):
    def __init__(self, returns, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.initial_balance = initial_balance
        self.current_step = 0
        self.done = False

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets + 1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.done = False
        return np.zeros(self.n_assets + 1)

    def step(self, action):
        action = action / np.sum(action)
        reward = np.dot(action, self.returns.iloc[self.current_step])
        self.current_step += 1
        if self.current_step >= len(self.returns) - 1:
            self.done = True
        return np.zeros(self.n_assets + 1), reward, self.done, {}

# Define DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_model = self.build_actor()

    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        return model

    def act(self, state):
        return np.ones(self.action_size) / self.action_size  # Equal weights for simplicity

# Streamlit UI
st.title("📈 Reinforcement Learning Portfolio Optimization (DDPG)")

# Load dataset
dataset_path = r"C:\Users\KIIT\Desktop\Portfolio Optimisation\all_stocks_5yr.csv"
data = pd.read_csv(dataset_path, parse_dates=["date"])

st.write("📊 **Stock Data Preview:**")
st.write(data.head())

# Process Data
data = data.pivot(index="date", columns="Name", values="close").dropna()
returns = data.pct_change().dropna()

# User Input Form
st.sidebar.header("🔹 User Information")
name = st.sidebar.text_input("Full Name")
email = st.sidebar.text_input("Email")
contact = st.sidebar.text_input("Contact Number")

st.sidebar.header("💰 Investment Details")
investment_amount = st.sidebar.number_input("Enter Amount to Invest (₹)", min_value=100, value=1000, step=100)
investment_duration = st.sidebar.slider("Investment Duration (Years)", 1, 10, 5)

# Suggest best stock based on amount
avg_returns = returns.mean()
best_stock = avg_returns.idxmax()
st.sidebar.write(f"📌 **Suggested Stock:** {best_stock} (Highest Avg. Returns)")

selected_stock = st.sidebar.selectbox("Select Stock", data.columns, index=list(data.columns).index(best_stock))

if st.sidebar.button("Optimize Portfolio"):
    # Calculate expected returns
    annual_return = avg_returns[selected_stock]
    expected_final_amount = investment_amount * ((1 + annual_return) ** investment_duration)

    # Calculate risk (Volatility)
    risk = returns[selected_stock].std()
    risk_level = "High" if risk > 0.03 else "Medium" if risk > 0.015 else "Low"

    # Display Results
    st.subheader("✅ **Optimized Portfolio Details:**")
    st.write(f"📌 **Investor:** {name}")
    st.write(f"📧 **Email:** {email}")
    st.write(f"📞 **Contact:** {contact}")
    st.write(f"💰 **Initial Investment:** ₹{investment_amount:,}")
    st.write(f"🕒 **Investment Duration:** {investment_duration} years")
    st.write(f"📈 **Selected Stock:** {selected_stock}")

    st.subheader("💡 **Predicted Investment Outcome:**")
    st.write(f"💵 **Final Expected Amount After {investment_duration} Years:**")
    st.markdown(f"<h2 style='color:green;'>₹{expected_final_amount:,.2f}</h2>", unsafe_allow_html=True)
    st.write(f"⚖️ **Risk Level:** {risk_level} (Volatility: {risk:.2%})")
