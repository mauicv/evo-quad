## evo-quad


### Notes:

- Model very quickly converges to actors that adopt static positions and stay in those positions. I think this is because the model is initially rewarded a lot for doing so. Moving around a lot before it knows how to walk results in the 100 point penalty cost whereas adopting a position just over the starting line and staying still at least gets it some reward.
  - reduce fall penalty
  - add oscillatory state inputs
