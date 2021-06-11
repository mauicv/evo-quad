## evo-quad


### Working Notes:

#### Static poses issue

Model very quickly converges to actors that adopt static positions and stay in those positions. I think the reason this happens for this model and not the openai ones is because here the model is outputting the positions of the motors rather than there torques. This we're much more likely to see static poses as each of those is at least a stable solution to the problem, especially if there is a fall penalty. Using torque means the network governs rate of change which means it's more likely to suggest dynamic solutions rather than static ones. This essentially injects noise into the model which acts as exploration.

Couple of potential solutions:
  - [x] Add oscillatory state inputs or nodes somewhere in the model. This should add some noise of the type we want as the model will be better able to convert oscillations into gate patterns than just the state inputs.
  - [x] Restrict action complexity. Currently there are 12 degrees of freedom, 4 legs * 3 joints in each. OpenAi ant env has 8. Supposing we reduce each action to moving the leg join one way or another this means there are 2^12 = 4096 options for each time step which is significantly more than the 2^8 = 256 for 8 variables. If we assume that the majority of the work done while walking is in the hip joints then we can restrict to just those joints and train on those 4, once the model has some suboptimal performance on them we can add the knee joins and so on.

Other things:
  - [X] Try different learning rates.
  - [x] Try different oscillation rates.
  - [X] Replace POSITION_CONTROL with VELOCITY_CONTROL
  - [X] Replace VELOCITY_CONTROL with TORQUE_CONTROL
  - [X] increase torque

Appears the torque applied to the motor wasn't strong enough which meant the default outcome was for the quadruped to fall over and be unable to get back up.

#### Alignment Issue:

Robot successfully learn to move forward. However it transpired that it had found a flaw in the simulation in order to do so. Possible solutions:

- [ ] Change motor control parameters
- [ ] Try smoothing network output, seems to be very volatile atm
- [ ] Add a zero zone, where if action is between -0.1 and 0.1 the torque is zero.

#### Network Output volatility

So the network outputs and extremely volatile signal. I checked to see if this was the same as in the ant case and it was. In fact evo-ants network is basically a very chaotic signal that's then squashed down onto [-1, 1] by the tanh output layer into essentially a square wave with different frequencies. I think our issue is that if a network output for a specific action is distributed around a mean sufficiently far outside of [-1, 1] then we end up with a signal that's mostly just takes value 1. The parameter updates are unlikely to move this signal by much and the end result will likely be that the model can't learn anything for that signal. This is the notable difference between ant and quad network outputs, namely that the mean of the network outputs is around 0 for the ant but outside of [-1, 1] for quad.

- [ ] Find a way to centre the output.
