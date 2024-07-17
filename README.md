# ThreePhaseFlow_Ag-MRDCVA
# Oil-gas-water three-phase flow process monitoring by Ag-MRDCVA
Source code of Ag-MRDCVA on oil-gas-water three-phase flow dataset.
The dataset is obtained through multiphase flow experiment at Tianjin Key Laboratory of Process Measurement and Control at Tianjin University.

The details of the model can be found in    
 [L. H. Li, et al. Manifold regularized deep canonical variate analysis with interpretable
attribute guidance for three-phase flow process monitoring, ESWA, 251, 124015, 2024.](https://doi.org/10.1016/j.eswa.2024.124015)

#### Notice: 
The code has been modified and improved.
The attribute prediction is realized by machine learning methods, including random forest, SVM, etc.
It makes the Ag-MRDCVA network less dependent on the shallow attribute prediction network and more lightweight.

#### Fast execution in command line:  
python3 AgMRDCVA.py      

#### Results Example: 
1. Identification for typical flow states, that is, testindex = 'data_ogw_test'

Data generating...   
Feature extracting...   
No training!   
Attribute predicting...   
Attribute prediction model: rf   
Identification of typical flow states...   
Overall accuracy：0.9125   
Identification accuracy for each flow state：[0.99  0.82  0.86  0.995 0.995 0.98  0.91  0.785 0.79  1.   ]   
   
2. Process monitoring for transition states, that is, testindex = 'data_transition_1' or testindex = 'data_transition_3'

Data generating...   
Feature extracting...   
No training!   
Attribute predicting...   
Attribute prediction model: rf   

#### All rights reserved, citing the following papers are required for reference:   
[1] L. H. Li, et al. Manifold regularized deep canonical variate analysis with interpretable
attribute guidance for three-phase flow process monitoring, ESWA, 251, 124015, 2024.
