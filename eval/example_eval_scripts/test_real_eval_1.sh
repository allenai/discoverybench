#!/bin/bash

query="What was the percentage increase in median wealth among black females from 1985 to 1990?"

gold_hypo="The largest increases in median wealth from 1985 to 1990 were among black females, with a percentage increase of 1385.71%.",

gold_workflow="Grouped the race and sex categories and calculated the percentage change in their median wealth."

gen_hypo="The percentage increase in median wealth among black females from 1985 to 1990 is approximately 1385.71%. Scientific Hypothesis: In the context of the provided dataset, which includes race, sex, and wealth variables from the National Longitudinal Study of Youth's 1979 cohort, it is hypothesized that there is a significant increase in median wealth for black females from 1985 to 1990. This hypothesis is supported by the data showing an approximate 1385.71% increase in median wealth for this demographic over the five-year period. Further investigation would be required to determine the cause of this increase and its statistical significance, as well as its implications on social and economic factors affecting black females during that time."

gen_workflow="1. Data Loading: Loaded the 'nls_incarceration_processed.csv' dataset using pandas from the specified path.
2. Data Filtering: Filtered the dataset to include only records for black females.
3. Median Wealth Calculation: Calculated the median wealth for black females for the years 1985 and 1990.
4. Percentage Increase Calculation: Computed the percentage increase in median wealth from 1985 to 1990 for black females."

metadata_path="/Users/bodhi/projects/datavoyager/DiscoveryBench/nls_incarceration_processed/metadata_14.json"

echo $(python3 ../discovery_eval.py --gold_hypo "$gold_hypo" --pred_hypo "$gen_hypo" --gold_workflow "$gold_workflow" --pred_workflow "$gen_workflow" --metadata_path "$metadata_path" --metadata_type "real" "$query")