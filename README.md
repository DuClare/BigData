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

## Data Analysis
### Overview

The eco2mix dataset offers a unique opportunity to explore France's energy landscape by analyzing regional and temporal variations in energy production and consumption. Our data analysis focused on uncovering key insights related to energy trends, seasonal fluctuations, and the role of renewable energy.
### Key Analyses and Findings
1. Monthly Energy Production and Consumption Trends

  - Aggregated energy production and consumption data by month and region, measured in GWh.
  - Identified seasonal trends: production consistently exceeds consumption, with peak periods observed during winter months.
  - Certain regions, like Auvergne-Rhône-Alpes and Normandie, emerge as energy exporters, while others, like Île-de-France, depend on imports to meet their needs.

2. Regional Energy Distribution

  - Bar charts of production and consumption highlighted disparities between regions:

    1. Regions with industrial infrastructure produce significantly more energy than they consume.
    2.  Dense metropolitan regions exhibit high consumption but lower production capabilities.

3. Daily Energy Consumption Patterns

  - Time-series analysis of daily energy consumption revealed:

    1. Peaks during winter and end-of-year holidays, driven by increased heating needs.
    2. Rolling monthly averages demonstrated gradual consumption trends over time.

4. Top Consumption Variations

  - Highlighted the 20 days with the largest day-to-day changes in energy consumption.
  - Observed that the most significant variations align with seasonal transitions and major national events.

5. Renewable Energy Surpass Analysis

  - Determined the first day each year (2013–2021) when cumulative renewable energy production surpassed consumption.
  - Renewable sources accounted for 25–33% of total energy production annually, with nuclear energy being a dominant contributor.

6. Energy Source Distributions

- Violin plots visualized the variability of energy production by source:

  1. Thermal and nuclear sources showed consistent high output, indicative of their role as baseload energy providers.
  2. Wind and solar production exhibited variability, reflecting their dependence on weather conditions.
  3. Hydraulic energy demonstrated a broad range, influenced by seasonal water availability.

### Visualizations

We created various visualizations to support our findings, including:
- Line plots for monthly trends.
- Bar charts comparing regional production and consumption.
- Time-series plots of daily consumption patterns.
- Violin plots showing energy source variability.

## Data Mining

In the data mining phase, we leveraged advanced techniques to uncover meaningful insights from the dataset. 

- **Clustering Analysis**: Using KMeans, we grouped French regions based on energy metrics, revealing distinct profiles such as renewable-heavy producers and consumption-dependent regions. 
- **Outlier Detection**: Anomalies in energy production and consumption were identified, uncovering unusual patterns tied to regional or seasonal factors. 
- **Predictive Modeling**: We built a robust LightGBM-based model to forecast energy consumption with high accuracy, highlighting key predictors like production metrics and temporal features. 

These analyses provide actionable insights to support efficient energy management and informed policy-making.
