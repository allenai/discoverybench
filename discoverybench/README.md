## Metadata Structure

The metadata provides info for understanding the datasets. Here's an overview of the structure:

- **id**: An identifier for the metadata.

- **domain**: The broad field of study or area of research.

- **workflow_tags**: A set of keywords summarizing the main processes or techniques used in the replication implementation. These tags provide an overview of the methodological approach, facilitating the identification of relevant analytical techniques.

- **domain_knowledge**:
  - Contextual information or insights related to the dataset, explaining how certain behaviors or variables can be interpreted within the field of study.
  - It helps open avenues to think in directions that an LLM might not have considered otherwise, broadening the understanding of the field.

- **datasets**: Contains detailed information about the datasets used, including:
  - **name**: The name or filename of the dataset.
  - **description**: A summary of the dataset's contents and the type of data it includes.
  - **columns**: Detailed descriptions of each column in the dataset, including:
    - **name**: The column's name or header.
    - **description**: Explanation of the data contained in the column and its significance.

## Synth Naming Convention

For the `synth` dataset, each directory within the train, dev, and test splits represents a unique **task dataset** constructed by varying the origin and target levels within the corresponding semantic tree. Specifically, we use the following naming convention: `{hyphenated-domain-name}_{start_height}_{target_depth}`, where `start_height` means the height (from leaves) of the observed nodes in the semantic tree, and `target_depth` means the depth (from root) of the target variable of the query hypothesis.
