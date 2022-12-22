from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
env.reset()

policy = lambda obs, agent: 0

for agent in env.agent_iter(1000):
    env.render()
    observation, reward, done, info = env.last()
    action = policy(observation, agent)
    env.step(action)
    env.close()