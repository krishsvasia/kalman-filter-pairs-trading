# Adaptive Kalman Filter
This file tracks both the evolution of the Kalman Filter, and  the logic behind each new feature as I work through its development
  
## Introduction to the Kalman Filter.
* A Kalman Filter is a recursive mathematical process, executed iteratively.
* The purpose of the Kalman Filter is to quickly estimate the true value of the object being measured, when the measured values contain noise (error, uncertainty).

![Kalman flowchart](assets/kalman_flowchart.png)

There are three main calculations:
1. Calculate the Kalman Gain.
1. Calculate the current estimate.
1. Calculate the new error in the estimate.

These calculations must be calculated iteratively in order for the Filter to improve its estimate and to approach the true value we are measuring.

### The Kalman Gain $K$
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
This is used when we want to track multiple variables, and places the Kalman Filter into a matrix format.  
A reminder of what the Kalman Filter does: takes an input from an observation. Takes a state of a particular situation. We receive information periodically. We want to keep updating where we think the value of the variable is at. We constantly receive measurements, and we make predications, and the Kalman Filter tells us which should hold greater value when calculating our estimate for each iteration.

### The process
1. Initial State - Contains a state matrix and a process covariance matrix, $X_0$ and $P_0$.
    * The state matrix $X_0$ typically contains the position and velocity of the value we are tracking in 1, 2, or 3 dimensions.
    * The process covariance matrix $P_0$ represents the error in the estimate / process.
1. As we iterate, the current state becomes the previous state, holding $X_{k-1}$ and $P_{k-1}$.
1. With a previous state, we are now able to calculate the prediction for the new state, $X_{k_p}$ and $P_{k_p}$.
    * $A$ and $B$ are adaptation matrices, and their purpose is to convert values into the correct format so that they can be used in the equations.
    * $X_{k_p} = AX_{k-1} + Bu_k + w_k$
        * This prediction makes use of the control variable matrix $u$, and we add on our prediction for how the control variables will affect the state matrix.
        * The predicted state noise matrix $w$ is also used to calculate the noise in that prediction.
    * $P_{k_p} = AP_{k-1}A^T + Q_k$
        * The process noise covariance matrix $Q$ accounts for any potential noise and needs to be accounted for in our prediction for the new process covariance matrix $P$.
1. After our prediction, we update it with the new measurement and the Kalman Gain to give us the updated state.
    * $H$ is also an adaptation matrix, like $A$ and $B$, and converts values into the correct format so that they can be used in the equations.
    * $Y_k = H_{k_m} + Z_k$
        * The measurement of the state $Y$.
        * As there may also be noise in the measurement, we need to add the measurement noise $Z$.
    * $K = \frac{P_{k_p}H^T}{HP_{k_p}H^T + R}$
        * The Kalman Gain decides how much we trust our estimate, and therefore what fraction of it we will use in our measurement and our prediction of the new state.
    * $X_k = X_{k_p} + K[Y-HX_{k_p}]$
1. Update the process covariance matrix $P$.
    * $P_k = (I - KH)P_{k_p}$
        * The matrix identity $I$.
    * $P$ is the error in the process of the Kalman Filter.
1. Output of the updated state.
    * The updated state matrix $X_k$
    * The updated process covariance matrix $P_k$
1. The process is then repeated.

### The State Matrix $X$
* The new state $X_k = AX_{k-1} + Bu_k + w_k$ consists of:
    * $X_{k-1}$ - previous state.
    * $u_k$ - control variable matrix.
    * $w_k$ - noise in the process.
    * $\Delta t$ - time for one cycle.

When calculating the movement of an object, we can use Newton's equations of motion.

#### Calculating $A X_{k-1}$
We multiply the previous state matrix by an adaptation matrix to put it into the correct format.

##### 1 Dimension
In one dimension, the state matrix will consist of a position, and a velocity.

* For position and velocity in the x direction:
$X = \begin{bmatrix} x \newline \dot{x} \end{bmatrix}$  
* For position and velocity in the y direction:
$X = \begin{bmatrix} y \newline \dot{y} \end{bmatrix}$

The adaptation matrix $A$ in one dimension.

$A = \begin{bmatrix} 1 \>\> \Delta t \newline 0 \>\>\>\> 1 \end{bmatrix}$

Multiplying $A$ by $X$: $AX = \begin{bmatrix} 1 \>\> \Delta t \newline 0 \>\>\>\> 1 \end{bmatrix}\begin{bmatrix} x \newline \dot{x} \end{bmatrix} = \begin{bmatrix} x + \Delta t \dot{x} \newline 0 + \dot{x} \end{bmatrix}$
* In the first row, we have the new position. This is calculated by adding the previous position to the distance moved (displacement = velocity * time).
* In the second row, we have the velocity.

##### 2 Dimensions
In two dimensions, the state matrix will consist of a position and velocity in the x direction, and a position and velocity in the y direction.

$X = \begin{bmatrix} x \newline y \newline \dot{x} \newline \dot{y} \end{bmatrix}$

#### Calculating $B u_k$
We multiply the control variable matrix by the adaptation matrix to get it into the correct format. $u_k$ represents the control variable, such as acceleration.

The $B$ matrix is derived from another equation of motion, $s = ut + \frac{1}{2} at^2$

$B = \begin{bmatrix} \frac{1}{2} \Delta t^2 \newline \Delta t \end{bmatrix}$

If there are no control variables active, $u_k = [0]$, or if it is acceleration, then $u_k = [a]$

## Transaction Cost Engine
Currently, my Kalman Filter isn't taking into account any friction values which would be faced in a live market environment. Therefore, while many of the trades which it makes seem profitable, in reality are not, as the profit is weighed down by factors such as commission and spread.

### Defining the Friction Values
The first step to implementing the TCE is to define the friction values which play against my trades, reducing profit:
* Commission - The fees charged by brokers to have my buy and sell orders made.
* Bid-Ask Spread - The difference between the lowest bid and the highest ask prices.
* Slippage - The amount which the market moves by while my order is being made.

### Setting the Friction Values
For the friction values, I will set them as realistic constants where I am unable to calculate them.

For the Bid-Ask Spread, the data I have doesn't offer the bid and ask prices, and so, while backtesting, I will assume a constant value of £0.01 per share.

The values I will use:
* Commission - £0.005 per share.
* Bid-Ask Spread - £0.01 per share.
* Slippage - 0.05% of the total trade price.

### The Logic
#### Equation Logic
After defining the friction values, I can now define the equation which calculates the cost of entering a trade.

As the spread is the difference between the bid and ask price, the transaction cost will only cover half the spread, as the price shown is the middle of this range.

$Transaction Cost = (Shares * Commission) + (Shares * \frac{Spread}{2}) + (Shares * sharePrice * Slippage/2)$

Final equation:
$$Transaction Cost = Shares * (Commission + \frac{Spread}{2} + sharePrice * Slippage)$$

#### Profit Logic
With this additional information, the expected value of all trades decreases, and some trades which may have seemed profitable, actually result in a loss. The Filter needs to be updated to account for this equation, keeping it in mind before making a trade.

##### Triggering the Engine
The equation must be triggered when my position changes, triggering once when entering or leaving a short or long position, and triggering twice when moving from a short position to a long position, or the other way around.

To calculate when my position changes, and by how much, I will need to compare my current_position with a new previous_position variable.

abs(current_position) + abs(previous_position)

This statement covers the logic of how many times I will count the equation when my position changes.

##### Calculating if a trade is profitable
As mentioned before, some trades which my project might have initially seemed profitable are no longer profitable. Next, I need to add logic to my code to account for this, plus an additional safety buffer.

$$ expectedVal > expectedCost * (2 + safetyBuffer)$$

The above statement will decide if a trade can be viewed as profitable. I am multiplying the expected cost by 2, to account for both the entry and exit cost, including a safety buffer. I will run the program with a safetyBuffer of different sizes, and compare the profit over the backtest to decide on the optimal value for the size and level of stocks which I am handling.

