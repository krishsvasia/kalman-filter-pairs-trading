# Autonomous Adaptive Kalman Agent
This research log tracks both the evolution of the Kalman Agent, and my learning process as I work through its development. Each entry represents a new milestone reached and its practical application to the agent.  
  
For the majority of this project, I will be learning from Michel Van Biezen's Kalman Filter playlist on Youtube.  
  
Having already produced part of the code for the Kalman agent, I will be spending the next few days working through concepts and explanations which have already been programmed, but I will be documenting my reflections from the videos to further my understanding and improve the code where possible.

## Introduction to the Kalman Filter.
* A Kalman Filter is a recursive mathematical process, executed iteratively.
* The purpose of the Kalman Filter is to quickly estimate the true value of the object being measured, when the measured values contain noise (error, uncertainty).

![Kalman flowchart](assets/kalman_flowchart.png)

There are three main calculations:
1. Calculate the Kalman Gain.
1. Calculate the current estimate.
1. Calculate the new error in the estimate.

These calculations must be calculated iteratively in order for the Filter to improve its estimate and to approach the true value we are measuring.

### Kalman Gain
The Kalman Gain is a "trust" factor, and places a relative importance on the estimate compared to the error in the data, placing more importance on the value with the lower error.

Formula:
* K = $\frac{E_{EST}}{E_{EST} + E_{MEA}}$ where $0\ge K \ge 1$

Meaning of the value:
* As the Kalman Gain approaches 1: measurements are more accurate, and estimates are unstable.  
* As the Kalman Gain approaches 0: measurements are inaccurate, and estimates are stable.  
  
As we iterate through the Kalman Filter, we expect the Kalman Gain to decrease as we approach the true value.

### Current Estimate
The current estimate is our guess in the current iteration for the value we are trying to calculate. It is equal to the previous estimate plus the Kalman Gain multiplied by the difference between the new measurement and the previous estimate.

Formula:
* $EST_t = EST_{t-1} + K(MEA - EST_{t-1})$

The current estimate is equal to the previous estimate plus the Kalman Gain multiplied by the difference between the new measurement and the previous estimate.

### Error in the Current Estimate
Formula:
* $^EEST_t = \frac{(E_{MEA})(^EEST_{t-1})}{(E_{MEA}) + (^EEST_{t-1})} \Rightarrow \> ^EEST_t = (1-K)(^EEST_{t-1})$

## The Multi-dimension Model