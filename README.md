## evo-quad


### Working Notes:

#### Static poses issue

Model very quickly converges to actors that adopt static positions and stay in those positions. I think the reason this happens for this model and not the openai ones is because here the model is outputting the positions of the motors rather than there torques. This we're much more likely to see static poses as each of those is at least a stable solution to the problem, especially if there is a fall penalty. Using torque means the network governs rate of change which means it's more likely to suggest dynamic solutions rather than static ones. This essentially injects noise into the model which acts as exploration.

Couple of potential solutions:
  - [x] Add oscillatory state inputs or nodes somewhere in the model. This should add some noise of the type we want as the model will be better able to convert oscillations into gate patterns than just the state inputs.
  - [x] Restrict action complexity. Currently there are 12 degrees of freedom, 4 legs * 3 joints in each. OpenAi ant env has 8. Supposing we reduce each action to moving the leg join one way or another this means there are 2^12 = 4096 options for each time step which is significantly more than the 2^8 = 256 for 8 variables. If we assume that the majority of the work done while walking is in the hip joints then we can restrict to just those joints and train on those 4, once the model has some suboptimal performance on them we can add the knee joins and so on.

Other things:
  - [ ] Try different learning rates.
  - [ ] Try different oscillation rates.

The behaviour of the model is very sensitive to the mutation rate. This means the improvements are likely to be very slow or very unstable.
