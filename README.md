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

In the data analysis phase, we explored the **eco2mix** dataset to uncover patterns and trends in energy production and consumption across France.

- **Seasonal Trends**:  
  Monthly energy production and consumption were analyzed, revealing:  
  - Peak consumption during winter months and holidays.  
  - Consistently higher production than consumption.  

- **Regional Disparities**:  
  - **Exporters**: Regions like **Auvergne-Rhône-Alpes** and **Normandie** produce more energy than they consume.  
  - **Importers**: Regions such as **Île-de-France** and **Provence-Alpes-Côte d’Azur** rely heavily on imports.  

- **Renewable Energy**:  
  - Identified the first day each year (2013–2021) when cumulative renewable production surpassed consumption.  
  - Renewable sources contributed 25–33% of total annual production.  

- **Source Variability**:  
  Violin plots highlighted differences in energy production:  
  - **Stable**: Nuclear and thermal energy provide consistent baseloads.  
  - **Variable**: Wind and solar depend heavily on weather conditions.  

This analysis offers actionable insights for optimizing energy distribution, increasing renewable investments, and preparing for seasonal demand spikes.

## Data Mining

In the data mining phase, we leveraged advanced techniques to uncover meaningful insights from the dataset. 

- **Clustering Analysis**: Using KMeans, we grouped French regions based on energy metrics, revealing distinct profiles such as renewable-heavy producers and consumption-dependent regions. 
- **Outlier Detection**: Anomalies in energy production and consumption were identified, uncovering unusual patterns tied to regional or seasonal factors. 
- **Predictive Modeling**: We built a robust LightGBM-based model to forecast energy consumption with high accuracy, highlighting key predictors like production metrics and temporal features. 

These analyses provide actionable insights to support efficient energy management and informed policy-making.
