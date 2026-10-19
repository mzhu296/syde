# Part 2: Gaussian Basis-Function Regression

## Part 2(e): Effect of the Number of Gaussian Basis Functions

Gaussian basis-function regression was fitted using the first 100 observations
as training data and the remaining 406 observations as test data. All 13 input
features were standardized using the training-set means and standard
deviations. The Gaussian basis functions were defined as

$$
\phi_j(\mathbf{x})=
\exp\left(-\frac{\|\mathbf{z}-\boldsymbol{\mu}_j\|^2}{2\sigma^2}\right),
$$

where $\mathbf{z}$ is the standardized input, $\boldsymbol{\mu}_j$ is a
standardized training observation selected as a centre, and $\sigma=1$. A
single random permutation with seed 572 was generated, and the first $K$
training observations in this permutation were used as centres for each value
of $K$. Thus, the centre sets were nested as the number of basis functions
increased. A bias term was included, and no regularization was applied.

| Gaussian basis functions | Training RMS | Testing RMS |
|---:|---:|---:|
| 5 | 5.193 | 9.848 |
| 15 | 4.293 | **9.792** |
| 25 | 3.847 | 9.811 |
| 35 | 3.663 | 9.914 |
| 45 | 3.244 | 9.817 |
| 55 | 3.061 | 9.864 |
| 65 | 2.858 | 9.846 |
| 75 | 1.908 | 11.935 |
| 85 | 0.764 | 11.798 |
| 95 | 0.299 | 11.755 |

Training RMS decreases monotonically as more Gaussian basis functions are
added, falling from 5.193 with five bases to 0.299 with 95 bases. This is
expected because each additional basis function adds a model parameter and
increases the model's ability to fit the training observations.

Testing RMS behaves differently. It remains close to 9.8 for models containing
5 through 65 basis functions, with the lowest observed value of 9.792 at 15
basis functions. Once the model reaches 75 basis functions, testing RMS rises
to approximately 11.9 even though training error continues to fall rapidly.
With 95 Gaussian bases plus the bias, the model has 96 adjustable weights for
only 100 training observations and nearly interpolates the training sample.
The widening gap between training and testing errors is therefore clear
evidence of overfitting. Without regularization, increasing the number of
Gaussian bases beyond roughly 65 improves memorization of the training data but
harms generalization.

The exact numerical curve depends on the randomly selected centres and the
Gaussian bandwidth. The fixed seed and stated bandwidth make the reported
experiment reproducible.

**Suggested Figure 5 caption:** Training and testing RMS error for
unregularized Gaussian basis-function regression as the number of randomly
selected training centres increases from 5 to 95. Training error decreases
monotonically, whereas testing error increases sharply beyond 65 basis
functions, demonstrating overfitting. The lowest observed testing RMS occurs
with 15 basis functions.

## Part 2(f): L2-Regularized Gaussian Basis Regression

A Gaussian regression model with 90 basis functions was evaluated using all 13
features and the first 100 observations. The regularization constant was
selected from $\{0,0.01,0.1,1,10,100,1000\}$ using 10-fold cross-validation.
The observations were shuffled reproducibly with random seed 572. Within each
fold, the feature means and standard deviations were calculated from the 90
fitting observations only. Those same 90 fitting observations were then used
as the Gaussian centres, ensuring that the 10 validation observations did not
influence normalization, centre construction, or weight estimation. The
Gaussian bandwidth was $\sigma=1$, and the bias coefficient was not penalized.

| $\lambda$ | Mean validation RMS | Standard error |
|---:|---:|---:|
| 0 | 3.999 | 0.487 |
| 0.01 | 3.840 | 0.440 |
| **0.1** | **3.731** | **0.426** |
| 1 | 4.026 | 0.442 |
| 10 | 4.998 | 0.412 |
| 100 | 5.635 | 0.401 |
| 1000 | 5.789 | 0.402 |

The lowest average validation RMS is 3.731 at $\lambda=0.1$, so this is the
selected regularization constant. Both $\lambda=0.01$ and $\lambda=0.1$
outperform the unregularized baseline, showing that modest coefficient
shrinkage improves generalization for the high-capacity 90-basis model. The
validation error begins increasing at $\lambda=1$ and continues to rise for
larger penalties. These stronger penalties shrink the Gaussian weights too
aggressively, increasing bias and causing underfitting. The minimum at
$\lambda=0.1$ therefore represents the best balance between the variance of
the nearly interpolating unregularized model and the bias introduced by strong
regularization.

**Suggested Figure 6 caption:** Mean 10-fold validation RMS for a 90-basis
Gaussian regression model plotted against the L2 regularization constant on a
logarithmic horizontal scale. Error bars show one standard error across folds,
and the dashed line represents the unregularized $\lambda=0$ baseline. A small
penalty improves generalization, with the minimum mean validation RMS occurring
at $\lambda=0.1$; larger penalties increasingly underfit the data.
