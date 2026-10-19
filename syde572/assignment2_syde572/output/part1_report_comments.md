# Part 1: Polynomial Basis-Function Regression

## Part 1(a): Model Complexity and Prediction Error

Polynomial basis-function regression was fitted using the first 100
observations as training data and the remaining 406 observations as test data.
Each input feature was standardized using the mean and standard deviation of
the training set. The design matrix contained powers of individual features
only; interaction terms such as $x_1x_2$ were not included. Models of degrees 1
through 7 were fitted using unregularized least squares.

| Polynomial degree | Training RMS | Testing RMS |
|---:|---:|---:|
| 1 | 2.0372 | 36.4449 |
| 2 | 1.5781 | 1,097.1883 |
| 3 | 1.4491 | 48,672.8178 |
| 4 | 1.2705 | 21,859,310.20 |
| 5 | 1.0966 | $1.836\times10^{10}$ |
| 6 | 1.0725 | $5.024\times10^{12}$ |
| 7 | 0.8815 | $6.879\times10^{15}$ |

Training RMS decreases continuously as the polynomial degree increases, while
testing RMS increases dramatically. This is strong evidence of overfitting.
With 13 input variables, a degree-$M$ model contains $1+13M$ weights. The
degree-7 model therefore has 92 weights but only 100 training observations,
allowing it to fit the training sample very closely.

The prescribed first-100 split is also not a randomized sample. Some test
observations contain feature values outside the ranges well represented in the
training set. High-degree, unregularized polynomials extrapolate extremely
poorly in these regions, producing the very large test errors shown in the
figure. Of the evaluated models, degree 1 has the lowest testing RMS, although
its relatively large error also indicates that the prescribed training sample
is not representative of the remaining observations.

**Suggested Figure 1 caption:** Training and testing RMS error for
unregularized polynomial basis-function regression of degrees 1 through 7. The
vertical axis uses a logarithmic scale because test error spans many orders of
magnitude. Training error decreases with model complexity, whereas testing
error grows rapidly, demonstrating severe overfitting and unstable
extrapolation.

## Part 1(b): Interpretation of the Degree-1 Weights

The degree-1 regression model was fitted using the same first 100 observations
and training-set normalization as in Part 1(a). Its prediction can be written
as

$$
\hat{y}=w_0+\sum_{j=1}^{13}w_j
\left(\frac{x_j-\mu_j}{\sigma_j}\right),
$$

where $w_0$ is the bias and each feature coefficient represents the change in
predicted `MEDV`, in thousands of dollars, associated with a one-training-
standard-deviation increase in that feature while the remaining variables are
held constant.

| Term | Weight |
|---|---:|
| Bias | 22.309000 |
| CRIM | -1.337974 |
| ZN | 0.122993 |
| INDUS | -0.531065 |
| CHAS | 0.000000 |
| NOX | 1.002651 |
| RM | **4.195634** |
| AGE | -2.166275 |
| DIS | **0.010715** |
| RAD | -0.193778 |
| TAX | -0.348631 |
| PTRATIO | 0.261264 |
| B | 0.181073 |
| LSTAT | 0.148761 |

In the assignment's feature-only indexing, the weight on the sixth feature is

$$
w_5=w_{\mathrm{RM}}=4.195634.
$$

One training standard deviation of `RM` is approximately 0.488 rooms. Thus, a
one-standard-deviation increase in `RM` raises predicted median value by about
4.196 thousand dollars. Expressed in the original input units, the fitted
effect is approximately $4.195634/0.488378=8.59$ thousand dollars per
additional room. The positive sign and comparatively large magnitude are
reasonable because larger dwellings generally have higher values.

The weight on the eighth feature is

$$
w_{\mathrm{DIS}}=0.010715.
$$

This corresponds to an increase of only about 0.011 thousand dollars for a
one-standard-deviation increase in weighted distance from the employment
centres. The estimated conditional linear effect is therefore effectively
zero. This is plausible because `DIS` is correlated with other location and
environment variables, so it may provide little independent information after
the other predictors have been included.

The coefficients should not be interpreted causally because the data are
observational and the predictors are correlated. Some apparently unexpected
signs can also result from multicollinearity and the unrepresentative first-100
training split. In addition, `CHAS` is zero for every training observation, so
its effect cannot be estimated from this training set and its fitted weight is
zero.

**Suggested Figure 2 caption:** Bias and standardized feature weights learned
by the degree-1 regression model. Positive coefficients are shown in green and
negative coefficients in red. `RM` has the largest positive feature weight,
whereas the coefficient of `DIS` is close to zero. The bias is displayed
separately so that the smaller feature weights remain visible.

## Part 1(c): One-Dimensional Polynomial Regression

To visualize polynomial regression directly, models were fitted using only
`ZN`, the second input feature and the proportion of residential land zoned for
lots larger than 25,000 square feet. The first 100 observations were used for
training, and `ZN` was standardized using the training-set mean and standard
deviation. Degrees 1, 3, and 7 were selected to illustrate low, moderate, and
high model complexity.

| Polynomial degree | Training RMS | Testing RMS |
|---:|---:|---:|
| 1 | 5.695 | 9.352 |
| 3 | 5.665 | 9.434 |
| 7 | 5.501 | 10.423 |

The degree-1 model produces a modest positive relationship between `ZN` and
median house value. However, observations with the same `ZN` value,
particularly `ZN = 0`, have a wide range of median values. This vertical spread
shows that `ZN` alone cannot explain most of the variation in `MEDV`.

The degree-3 model introduces mild curvature but does not improve
generalization. Its training RMS decreases slightly from 5.695 to 5.665, while
its testing RMS increases from 9.352 to 9.434. The added terms therefore fit
the training observations somewhat more closely without improving predictions
for unseen data.

The degree-7 polynomial oscillates substantially, especially in regions with
few training observations. Its training RMS falls to 5.501, but its testing
RMS rises to 10.423. This widening generalization gap is evidence of
overfitting: the higher-order model uses unnecessary curvature to follow the
limited training sample and consequently performs worse on the test set. Among
these three one-dimensional models, the degree-1 polynomial has the lowest
testing error and provides the most defensible representation of the
relationship.

**Suggested Figure 3 caption:** Training observations, test observations, and
learned one-dimensional polynomial regression curves using the standardized
`ZN` feature. Increasing the degree reduces training error slightly but raises
testing error and produces an increasingly oscillatory fitted curve,
demonstrating overfitting.

**Suggested overlay figure caption:** Overlay of the degree-1, degree-3, and
degree-7 polynomial regression curves using the standardized `ZN` feature. The
degree-1 and degree-3 predictions remain comparatively smooth and similar,
whereas the degree-7 curve oscillates strongly between sparsely represented
input values. The overlay makes the high-degree model's increased variance and
poorer test performance directly visible.

## Part 1(d): L2-Regularized Polynomial Regression

A degree-8 polynomial model was fitted using only `ZN` and the first 100
observations. The regularization constant was selected from
$\{0,0.01,0.1,1,10,100,1000\}$ using 10-fold cross-validation. The observations
were shuffled reproducibly with random seed 572 before being divided into ten
equal folds. Within each fold, the `ZN` normalization parameters were estimated
using only the 90 fitting observations and then applied to the 10 validation
observations. The bias coefficient was not regularized.

| $\lambda$ | Mean validation RMS | Standard error |
|---:|---:|---:|
| 0 | 9.985 | 4.135 |
| **0.01** | **6.586** | **0.907** |
| 0.1 | 7.674 | 1.906 |
| 1 | 9.344 | 3.474 |
| 10 | 10.251 | 4.279 |
| 100 | 9.792 | 3.803 |
| 1000 | 7.422 | 1.518 |

The smallest average validation error is 6.586 at $\lambda=0.01$, so this is
the selected regularization constant. Compared with the unregularized baseline
RMS of 9.985, a small amount of L2 regularization substantially improves
validation performance. This indicates that the degree-8 polynomial has high
variance when its coefficients are unconstrained. A penalty of 0.01 shrinks
the polynomial coefficients enough to reduce this instability without imposing
the stronger bias associated with the larger candidate penalties. Validation
error is not strictly monotonic in $\lambda$ because it is averaged across a
small dataset whose folds differ considerably, as reflected by the error bars.

**Suggested Figure 4 caption:** Mean 10-fold validation RMS for degree-8
polynomial regression using `ZN`, plotted against the L2 regularization
constant on a logarithmic horizontal scale. Error bars show one standard error
across folds, and the dashed horizontal line shows the unregularized
$\lambda=0$ baseline. The minimum mean validation error occurs at
$\lambda=0.01$.
