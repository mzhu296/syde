## Part 1(c): One-Dimensional Polynomial Regression

Figure X compares polynomial regression models of degrees 1, 3, and 7 using
only `ZN`, the proportion of residential land zoned for lots larger than
25,000 square feet. The models were trained on the first 100 observations. The
input was standardized using the mean and standard deviation of the training
set, and the same transformation was then applied to the test observations.

The degree-1 model produces a modest positive relationship between `ZN` and
median house value. Its training and testing RMS errors are 5.695 and 9.352,
respectively. However, observations with the same value of `ZN`, particularly
`ZN = 0`, have a wide range of median house values. This vertical spread shows
that `ZN` alone cannot explain most of the variation in `MEDV`.

The degree-3 model introduces mild curvature, but it does not improve
generalization. Its training RMS decreases slightly to 5.665, while its testing
RMS increases to 9.434. The additional polynomial terms therefore fit the
training observations somewhat more closely without improving predictions for
unseen data.

The degree-7 polynomial is substantially more flexible and oscillates strongly,
especially in regions containing relatively few training observations. Its
training RMS falls to 5.501, but its testing RMS rises to 10.423. This widening
gap is evidence of overfitting: the higher-order model uses unnecessary
curvature to follow the limited training sample and consequently performs worse
on the test set. Of these three models, the degree-1 polynomial has the lowest
testing error and provides the most defensible representation of the
one-dimensional relationship.

| Polynomial degree | Training RMS | Testing RMS |
|---:|---:|---:|
| 1 | 5.695 | 9.352 |
| 3 | 5.665 | 9.434 |
| 7 | 5.501 | 10.423 |

**Suggested figure caption:** Training observations, test observations, and
learned one-dimensional polynomial regression curves using the normalized `ZN`
feature. Increasing polynomial degree reduces training error but increases test
error and produces increasingly oscillatory fits, demonstrating overfitting.
