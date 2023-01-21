---
layout: post
comments: true
title:  "Correlation "
excerpt: "Note on population correlation, sample correlation and spurious sample correlation. With python code"
date:   2020-06-07 00:30:00
mathjax: true

---

Correlation refers to the degree to which two random variables are *linearly* related.  The correlation reflects the strength and direction of a linear relationship, but not the slope of that relationship.
The correlation does not depend on the units of measurement and is always between -1 and 1. The correlation is +1 in the case of a perfect direct (increasing) linear relationship, -1 in a prefect inverse (decreasing) relationship and 0 when there is no linear relationship.

## Population correlation   
The population (Pearson's product moment) correlation coefficient between two random variables $$X$$ and $$Y$$ is represented by the symbol $$\boldsymbol{\rho_{X,Y}}$$

$$
\begin{align}
    \rho_{X,Y} = Corr(X,Y)=\frac{Cov(X,Y)}{\sigma(x)\sigma(y)} \tag{1}
  \end{align}
$$  


Independence:  

$$
  \begin{align}
    X\text{ and }Y\text{ are independent }=> \rho_{X,Y}=0 \tag{2}\\   
    \rho_{X,Y}=0\ne>X\text{ and }Y\text{ are independent!}   
  \end{align}
$$


## Sample correlation
The sample correlation between $$x_i$$ and $$y_i$$, indexed by $$i=1,...,n$$ is represented by the symbol $$\boldsymbol{r_{x_iy_i}}$$. It can be used to *estimate* the population correlation between $$X$$ and $$Y$$.   

$$\begin{align}
  r_{xy}&=\frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{(n-1)s_xs_y} \tag{3}\\   
\\
  &=\frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^n(x-\bar{x})^2\sum_{i=1}^n(y-\bar{y})^2}}
  \end{align}
  $$   

where $$\bar{x}$$ and $$\bar{y}$$ are the sample means and $$s_x$$ and $$s_y$$ are the *corrected sample standard deviations* of $$X$$ and $$Y$$.   $$r^2$$ is the proportion of the total variance ($$s_y^2$$) of $$y$$ that can be explained by the linear regression of $$y$$ on $$x$$.

## Spurious Correlation
Sample (observed) correlation $$r$$, just like sample mean etc, doesn't necessary correspond to the population (real) correlation $$\rho$$ !   
**Never assume correlations unless there is statistical significance.**

## Python Experiments
### Import libraries
```python
import numpy as np
import pandas as pd
import scipy.stats as stat
import plotly.graph_objs as go
from plotly.subplots import make_subplots
```


### Generate two random variables with correlation coefficient $$r$$:
We can use a simplified bivariate form of [Cholesky decomposition](wikipedia.org/wiki/Cholesky_decomposition), sometimes called the Kaiser-Dickman algorithm (Kaiser & Dickman, 1962) to create two random variables with correlation coefficient $$r$$.   
First we create two random variables $$X_1$$ and $$X_2$$ with $$var(X_1) = var(X_2)$$.   
Then we apply the following Kaiser-Dickman algorithm to create correlated random variables $$X$$ and $$Y$$:

$$\begin{align}
  \begin{cases}
    X = X_1\\
    Y = r*X_1 + \sqrt{1-r^2}*X_2 \tag{4}
  \end{cases}
\end{align}$$

```python
def generate_rv(size=18, r=0):
    '''Generates 2 random variables with correlation coefficent ~ cc'''
    X1 = stat.norm.rvs(size=size)
    X2 = stat.norm.rvs(size=size)
    X = X1
    Y = r*X1 + np.sqrt(1-r**2)*X2
    return X, Y
```
### Visualization of random viariables with different correlation coefficients:   
```python
target_r = [-1, -.7, -.4, -.1, 0, .1, .4, .7, 1]
traces, real_r = [], []
for i, r in enumerate(target_r):
    X, Y = generate_correlated_rvs(1500, r)
    real_r.append(stat.pearsonr(X,Y)[0])
    traces.append(go.Scatter(mode='markers',x=X, y=Y))
fig = make_subplots(rows=3, cols=3,
                    column_widths=[1/3, 1/3, 1/3], row_heights=[1/3, 1/3, 1/3],
                    subplot_titles=[f'real r = {r:+.2f}' for r in real_r])
fig.update_layout(title='Random variables with different correlation coefficient r',
                  width=750, height=750, showlegend=False)
fig.update_traces(marker=dict(size=2))
for i, trace in enumerate(traces):
    fig.add_trace(trace, row=i // 3 + 1, col=i % 3 + 1)
    fig['layout'][f'xaxis{i+1}'].update(range=[-5, 5])
    fig['layout'][f'yaxis{i+1}'].update(range=[-5, 5])
fig   
```
>![Random variables with different correlation coefficient r](/assets/2020-06-07-correlation/plot_rv_different_r.png)

### Same correlation coefficient, but different slope:
```Python
X1 = stat.norm.rvs(size=1500)
X2 = stat.norm.rvs(size=1500)
r = 1
traces, real_r = [], []
for a in [1, .5, .1, -.1, -.5, -1]:
    X = X1
    Y = a*r*X1 + np.sqrt(1-r**2)*X2
    real_r.append(stat.pearsonr(X,Y)[0])
    traces.append(go.Scatter(mode='markers',x=X, y=Y))
fig = make_subplots(rows=2, cols=3,
                    column_widths=[1/3, 1/3, 1/3], row_heights=[1/2,  1/2],
                    subplot_titles=[f'real r = {r:+.2f}' for r in real_r])
fig.update_layout(title='Random variables with different slope',
                  width=750, height=500, showlegend=False)
fig.update_traces(marker=dict(size=2))
for i, trace in enumerate(traces):
    fig.add_trace(trace, row=i // 3 + 1, col=i % 3 + 1)
    fig['layout'][f'xaxis{i+1}'].update(range=[-5, 5])
    fig['layout'][f'yaxis{i+1}'].update(range=[-5, 5])
fig
```
>![Same correlation coefficient, but different slope](/assets/2020-06-07-correlation/plot_rv_different_slope.png)



### Spurious sample correlations
Create 2 un-correlated random variables $$X$$ and $$Y$$:
```python
X = stat.norm.rvs(size=1500)
Y = stat.norm.rvs(size=1500)
 ```
Check if the (real) correlation $$Corr(X,Y) \approx 0$$
```Python
rho = stat.pearsonr(X,Y)[0]
print(rho)
```
>-5.875043840207288e-05   

X and Y are un-correlated as expected. Now, let's take 1.000 different samples $$x$$ and $$y$$ of size 15 from $$X$$ and $$Y$$ respectively and calculate their sample correlations.
```python
sample_corrs = pd.DataFrame()
for i in range(1000):
    x = np.random.choice(X,15)
    y = np.random.choice(X,15)
    sample_corrs = sample_corrs.append({'i':i, 'r':stat.pearsonr(x,y)[0]},
                                       ignore_index=True)
print(sample_corrs.head())
```
>| |    i|         r|
>|-|-----|----------|   
>|0|  0.0| -0.052040|   
>|1|  1.0| -0.060840|   
>|2|  2.0| -0.099591|   
>|3|  3.0| -0.232984|  
>|4|  4.0|  0.214982|   

```python
fig = go.Figure()
fig.add_trace(go.Bar(x=sample_corrs['i'],y=sample_corrs['r'],marker_color='red',
                     marker_line_width=0))
fig.update_layout(width=500, height=500,
                  yaxis_range=[-1,1])
```
>![Spurious correlations](/assets/2020-06-07-correlation/plot_sample_corr_15.png)

Spurious correlations are all over the place !
Now, let's do the same but with samples of size 45
```python
sample_corrs = pd.DataFrame()
for i in range(1000):
    x = np.random.choice(X,45)
    y = np.random.choice(X,45)
    sample_corrs = sample_corrs.append({'i':i, 'r':stat.pearsonr(x,y)[0]},
                                       ignore_index=True)
fig = go.Figure()
fig.add_trace(go.Bar(x=sample_corrs['i'],y=sample_corrs['r'],marker_color='red',
                     marker_line_width=0))
fig.update_layout(width=500, height=500,
                 yaxis_range=[-1,1])
```
>![Spurious correlations](/assets/2020-06-07-correlation/plot_sample_corr_45.png)

Correlations are more compressed around zero, but still all over the place!
