#!/bin/bash

query="How strongly does BA degree completion vary with socioeconomic status?"

gold_hypo="Socioeconomic status (SES) is a significant predictor of BA degree completion. SES has a positive relationship with college degree completion with a coefficient of 0.4729.",

gold_workflow="OLS Regression"

gen_hypo="The analysis of the relationship between socioeconomic status (SES) and BA degree completion using \
    logistic regression suggests that higher socioeconomic status is associated with an increased likelihood of completing \
    a BA degree. However, the model's limitations in predicting BA degree completions accurately indicate that SES alone \
    may not fully explain BA degree completion rates. The formulated hypothesis suggests that while there is a positive \
    relationship between SES and BA degree completion, additional factors not captured in this analysis likely influence \
    this outcome. Further research incorporating more variables and possibly more complex models is recommended to fully \
    understand the determinants of BA degree completion."

gen_workflow="1. **Load the Dataset**: The dataset nls_ses_processed.csv was successfully loaded, providing an \
    overview of its structure, including relevant variables like SES and BA DEGREE COMPLETED.
    2.**Data Exploration**: The dataset was explored to understand its structure, revealing no missing values in the key \
    variables of interest (SES and BA DEGREE COMPLETED) and providing summary statistics.
    3. **Data Preprocessing**: It was determined that no extensive preprocessing was needed beyond the initial exploration, \
    as there were no missing values in the key variables.
    4. **Statistical Analysis**: Logistic regression was performed to analyze the relationship between socioeconomic status \
    (SES) and BA degree completion (BA DEGREE COMPLETED). The analysis revealed a positive coefficient for SES, \
    suggesting an increase in the likelihood of completing a BA degree with higher socioeconomic status. However, \
    the models predictive performance was limited, particularly in predicting BA degree completions.
    5. **Hypothesis Formulation**: Based on the statistical analysis, a hypothesis was formulated stating that while the \
    likelihood of completing a BA degree increases with higher socioeconomic status, SES alone may not be a sufficient \
    predictor of BA degree completion, indicating the need for further research incorporating additional variables."

metadata_path="../../DiscoveryBench/nls_ses_processed/metadata_1.json"

echo $(python3 ../discovery_eval.py --gold_hypo "$gold_hypo" --pred_hypo "$gen_hypo" --gold_workflow "$gold_workflow" --pred_workflow "$gen_workflow" --metadata_path "$metadata_path" --metadata_type "real" "$query")