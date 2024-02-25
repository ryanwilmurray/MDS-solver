# Versions of Marginal Minimization

This package contains various iterations of our marginal minimization method, to facilitate experimentation of various ideas and techniques.
The `default` module contains the 'production' version, and various other branches of the method will be in other modules named accordingly.

## Usage
In each module the method will be contained in a class called `MM`, the parameters of which will be documented for each version.
```python
# Importing various versions
import ndsolver.versions.default as v1
import ndsolver.versions.experimental as v2

# Utilizing the versions in parallel
minimizer1 = v1.MM(...)
minimizer2 = v1.MM(...)
```