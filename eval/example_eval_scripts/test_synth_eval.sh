#!/bin/bash

query="Is there a relationship between the average number of tourist visits per year and the material usage ratio in historic structures?"

gold_hypo="Structures with a combination of lower average tourist visits per year, higher material durability scores, being in a historical conservation area, and not being used for ceremonial purposes tend to have a higher material usage ratio, possibly due to better material quality and preservation efforts in historical areas."

gold_workflow=""

gen_hypo="The scientific hypothesis generated from the analysis of the provided dataset on ancient languages suggests that there is a positive linear relationship between the translation effectiveness score for deciphering ancient scripts and the ratio of the number of digitized artifacts to the number of available deciphering tools, adjusted by the complexity level of the script and the level of educational support and media recognition for the language. Specifically, the linear coefficient describing this relationship is approximately 0.43. This implies that, for each unit increase in the adjusted ratio of digitized artifacts to deciphering tools, there is an average increase of 0.43 units in the translation effectiveness score, suggesting that the availability of digitized artifacts, deciphering tools, and educational"

gen_workflow=""

metadata_path="../../DiscoveryBenchSynth/benchmark/ancient-architecture_0_0/metadata_0.json"

echo $(python3 ../discovery_eval.py --gold_hypo "$gold_hypo" --pred_hypo "$gen_hypo" --gold_workflow "$gold_workflow" --pred_workflow "$gen_workflow" --metadata_path "$metadata_path" --metadata_type "synth" "$query")