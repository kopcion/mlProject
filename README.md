# LearningFromHumanPreferences

## Based on the [paper](https://arxiv.org/pdf/1706.03741.pdf)
Deep Reinforcement Learning from Human Preferences


## Setup instructions
1. install openai gym and mujoco
https://github.com/openai/mujoco-py
  ```pip install gym```

2. install http server to serve video files
  ```npm install http-server -g```
3. install frontend dependencies
  ```cd frontend && npm install```
4. go to backend videos directory and start http server
  ```cd backend/videos && npx http-server -c-1 --cors```
5. start frontend
  ```cd frontend && npm start```
6. start backend
  ```cd backend && python main.py```
7. app should be available at http://localhost:3000/training


## Used dependencies
Using TRPO implementation from https://github.com/ikostrikov/pytorch-trpo
