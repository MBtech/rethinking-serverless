# Performance Data
This directory contains the performance data collected by running the serverless functions on OpenFaaS in AWS.
Performance data of each serverless benchmark is located in respective directories. 

The naming conventions used for the csv files is: 
<benchmark>_<input>_<memory_share>Mi_<cpu_share>m_<instance_type>.csv

We use the response-delay in the CSV file as the metric for serverless functions' execution time (if the status code is 200). 