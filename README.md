# Evaluation of Interpretable AI on Synthetic Data With Interactions

Chapter 5 of Johnny Joyce's PhD thesis. Based on *"On Polynomial Threshold Functions of Bounded Tree-Width"* by Karine Chubarian, Johnny Joyce, and György Turán.

*Festschrift for Johann A. Makowsky*. Springer NatureSwitzerland, 2024.


## Abstract

> The tree-width of a multivariate polynomial is the tree-width of the hypergraph with hyperedges corresponding to its terms. Multivariate polynomials of bounded tree-width have been studied by Makowsky and Meer as a new sparsity condition that allows for polynomial solvability of problems which are intractable in general. We consider a variation on this theme for Boolean variables. A representation of a Boolean function as the sign of a polynomial is called a polynomial threshold representation. We discuss Boolean functions representable as polynomial threshold functions of bounded tree-width. Applications are given to two problems for Bayesian network classifiers, addressing questions in machine learning interpretability from theoretical and experimental aspects. We also give a separation result between the representational power of positive and general polynomial threshold functions.

## Description

* `Experiment - polynomials from paper.ipynb` : Jupyter notebook for running the experiment on a GA<sup>2</sup>M with the exact tree augmented naive Bayes (TAN) and corresponding conditional probabilities.
* `Experiment - miscellaneous functions.ipynb` : Other experiments that were not presented in the paper
* `networks.py` : Contains helper functions for main notebooks

## License

MIT License

Copyright (c) 2024 Johnny Joyce

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
