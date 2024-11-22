# Classification Models Comparison: Glassbox vs. Blackbox

## Introduction

This project aims to build and compare classification models for predicting customer satisfaction based on a dataset of airline service and customer feedback. The models will include both **glassbox models** (e.g., decision trees, logistic regression) and **blackbox models** (e.g., neural networks, random forests) to analyze trade-offs in interpretability, accuracy, and usability.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Project Goals](#project-goals)
- [Installation](#installation)
- [Contributors](#contributors)
- [License](#license)

## Dataset Overview

The dataset contains 103,904 rows and 25 columns. It includes customer demographics, travel details, service ratings, and a satisfaction label. A summary of the dataset is as follows:

- **Target Variable**: `satisfaction` (binary: `satisfied` or `neutral or dissatisfied`)
- **Features**:
  - Customer demographics (e.g., `Gender`, `Age`, `Customer Type`)
  - Travel details (e.g., `Type of Travel`, `Class`, `Flight Distance`)
  - Service ratings (e.g., `Inflight wifi service`, `Cleanliness`, `Baggage handling`)
  - Delay metrics (e.g., `Departure Delay in Minutes`, `Arrival Delay in Minutes`)
- **Missing Data**: Minimal missing values in the `Arrival Delay in Minutes` column.

### Sample Data

| Gender | Customer Type | Age | Type of Travel | Class    | Flight Distance | Satisfaction       |
|--------|---------------|-----|----------------|----------|-----------------|--------------------|
| Male   | Loyal Customer| 13  | Personal Travel| Eco Plus | 460             | neutral or dissatisfied |
| Male   | Disloyal Customer| 25 | Business travel| Business | 235             | neutral or dissatisfied |
| Female | Loyal Customer| 26  | Business travel| Business | 1142            | satisfied          |

## Project Goals

1. **Data Exploration**:
   - Understand patterns and correlations in customer satisfaction data.
2. **Model Development**:
   - Train and evaluate both glassbox (interpretable) and blackbox (highly predictive) models.
3. **Comparison**:
   - Compare models on accuracy, interpretability, and practical use cases.
4. **Insights**:
   - Generate actionable insights for improving airline service.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo


## Contributors

- Woojin Heo
- Abhiraj Singh
- Bhavna
- Shlok Sudhir Kamat

