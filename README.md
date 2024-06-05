Heterogeneous neighbor contrastive graph attention network(HNCGAT)
====
This repository contains the author's implementation in Pytorch for the paper "HNCGAT: heterogeneous neighbor contrastive graph attention network for metabolite-protein interaction prediction in plant".

Environment Requirement
===
The code has been tested running under Python 3.9.13. The required packages are as follows:

•	python == 3.9.13

•	pytorch == 1.10.1

•	numpy == 1.22.4

•	scipy == 1.8.1

•	sklearn == 1.0.2


Usage
===
"HNCGAT.py" is the implementation of the HNCGAT model. 

"/src/dataset/" contains the dataset required for running the HNCGAT model.

"/src/result/" contains the result after running the HNCGAT model. And the protein and metabolite embbedings learned by the HNCGAT model.

"/src/model/" contains the learned weight model trained by the author.


