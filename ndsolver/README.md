## Goals

1. Perform analysis on simpler cases which we have intuition for the optimal embedding
2. Create a framework for "unit testing" different methods with these base cases
3. Gain a qualitative understanding of features of qMDS and behaviours for changes in hyperparameters
4. Derive closed-form forumlas in terms of $X$ and 
5. Optimize parameters for both numerical and visual results, reach better results than PCA and MDS
6. Optimize time and space efficiency of algorithm, reach a comparable runtime to PCA


## Notes

- Marginal minimization appears to be non-monotonic and unstable by default, but does improve cost over iterations on average
    - It usually converges (relatively quickly) to jumping between two states of largely differing costs
    - It also usually alternates between two significantly differing scales
- The method only somewhat correlates to global minimization (it currently deviates from the perfect embedding)
- Accuracy and stability was able to be introduced by batching elements where $|Y_i| > \epsilon$, for some sufficiently sized threshold $\epsilon$.
    - The batched elements then updated normally
    - A single step of traditional descent is performed on the other elements, with an arbitrarily selected $h$ and an un-normlaized gradient (haven't experimented much with the effects of normalization here)
    - So far, we have only arbitrarily selected $\epsilon$, and have not tried deriving a closed-form formula
- The `quartic_initialization_vectorized` function is currently handling similiar eigenvalues and negative $k$ values in a somewhat arbitrary way, this can likely be improved to something intuitive and adaptive, hopefully with improved results


## Questions

- How much accuracy is lost via updating all components at once?
- How much can batching mitigate the effects of this inaccuracy?
- Is it possible for a marginal method to converge to or at least accept the global minimizer?


## Ideas

- Maybe capping of negative eigenvalues of $\psi$ at $0$ is causing instability
- In order for MM to be stable and converge, it probably needs to be taking smaller steps each iteration
    - Should probably try a decreasing learning rate schedule, ie $Y = Y + h \cdot (Y_{new} - Y)$
    - Not sure how this would affect the accuracy of the result, but it would at least force convergence