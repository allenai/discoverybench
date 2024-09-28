#!/bin/bash

query="How did the wealth levels of individuals with a history of incarceration compare to those never incarcerated in 1996?"

gold_hypo="Individuals with a history of incarceration in 1996 have lower wealth levels compared to those never incarcerated",

gold_workflow=""

gen_hypo="The t-test results show a t-statistic of approximately -9.72 and a p-value of 7.26e-20, which is much lower than the conventional alpha level of 0.05. This indicates, in 1996, individuals with a history of incarceration had significantly lower average wealth levels than those who were never incarcerated, as supported by the statistical significance found in the t-test."

gen_workflow=""

metadata_path="/Users/bodhi/projects/datavoyager/DiscoveryBench/nls_incarceration_processed/metadata_1.json"

echo $(python3 ../discovery_eval.py --gold_hypo "$gold_hypo" --pred_hypo "$gen_hypo" --gold_workflow "$gold_workflow" --pred_workflow "$gen_workflow" --metadata_path "$metadata_path" --metadata_type "real" "$query")