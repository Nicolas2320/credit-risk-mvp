# End-to-End Credit Risk Modeling & Strategy Simulation

## Overview
This project implements an end-to-end credit risk analytics framework to estimate the Probability of Default (PD) for retail loan applicants and translate model outputs into actionable credit decisions. The solution covers the full lifecycle from data ingestion and feature engineering to modeling, calibration, score banding, and portfolio-level strategy simulation.

## Business Problem
Retail lenders must balance portfolio growth with credit risk. Approval decisions are typically made at origination using risk scores or PD estimates. This project focuses on building a calibrated PD model and a transparent scorecard framework that supports approval policy design, portfolio monitoring, and risk strategy evaluation.

## Data & Architecture
The project uses a synthetic dataset inspired by the Home Credit Default Risk portfolio from Kaggle, including application-level information and aggregated bureau data.  
A cloud-native AWS architecture was implemented:
- Amazon S3 for raw, curated, and model output data
- AWS Glue for ETL and feature engineering
- Amazon Athena for data validation and analytical queries
- Amazon SageMaker for model development and evaluation
- Power BI for portfolio monitoring and strategy simulation

An end-to-end architecture diagram is provided in the `architecture/` folder.

## Modeling & Calibration
Multiple classification models were evaluated, including Logistic Regression, Support Vector Machines, and Random Forests. Logistic Regression was selected as the production model due to its interpretability, stability, and alignment with industry best practices in credit risk modeling.

Raw model outputs were calibrated using Platt Scaling to ensure predicted probabilities align with observed default rates. Calibration quality was assessed using Brier Score and calibration curves.

## Scorecard & Credit Strategy
Calibrated PDs were transformed into a monotonic risk score and five risk bands (A–E). These bands enable transparent approval policies and portfolio segmentation.  
A credit strategy simulator was developed to evaluate approval policies (example: approve A–C vs A–D) and quantify trade-offs between approval rate and expected defaults.

## Dashboard
Interactive Power BI dashboards provide portfolio-level and segment-level insights, including:
- PD distribution and risk band composition
- Approved vs rejected populations under different strategies
- Risk concentration across segments

Dashboard overview are available in `dashboard/screenshots/`.

## Key Results
- AUC ≈ 0.74 with stable validation performance
- Well-calibrated PDs across score bands
- Approving Bands A–C yields ~59% approval rate with average PD ≈ 3.7%
- Expanding to Band D increases approval but materially raises expected defaults

## Repository Structure
See folder structure for report, notebooks, architecture, and dashboard artifacts.

## Next Steps
- Integrate additional behavioral datasets (e.g., previous applications, repayment history)
- Evaluate more complex models (e.g., gradient boosting) with explainability constraints
- Extend the framework to expected loss modeling (PD × LGD × EAD)

## Disclaimer
This project uses synthetic data inspired by real-world credit portfolios. All results are illustrative and intended for educational and portfolio demonstration purposes only.
