# Big Data Project

## Dataset

We used the **eco2mix** dataset, which provides data on energy production and consumption across different regions of France at various dates.

### Why did we choose this dataset?

This dataset is particularly interesting because it allows us to:
- Identify which regions produce the most energy and explore the reasons behind it.
- Analyze which dates correspond to higher electricity consumption in France.
- Study the correlation between energy production and consumption patterns.

## Data Cleaning

When we received this dataset, we encountered several issues:
- Many columns contained **NA** values and unnecessary information. 
- To clean the data, we took the following steps:
  - Removed columns deemed irrelevant to our analysis.
  - Deleted rows with too many **NA** values, as they provided no useful information.
  - For missing values in energy production data across different sectors, we chose to fill them with `0` instead of using the mean, as averaging would not make sense in this context.
- Added new calculated columns, such as:
  - **TCO** (Total Consumption Over Time) to provide a cumulative view of energy usage.
  - **Production** to aggregate energy production from various sources for a more comprehensive analysis.