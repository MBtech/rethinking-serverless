# Performance Data
This directory contains the performance data collected by running the serverless functions on OpenFaaS in AWS.
Performance data of each serverless benchmark is located in respective directories. 

The naming conventions used for the csv files is: 
<benchmark>_<input>_<memory_share>Mi_<cpu_share>m_<instance_type>.csv

Benchmarks: `faceblur`, `facedetect`, `ocr`, `linpack`, `transcode`, `s3`
Input: `See input/README.md`
Memory Share: `128`, `256`, `512`, `768`, `1024`, `2048`
CPU share: `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`
Instance type: `c6g`, `m6g`, `c5`, `m5`, `c5a`, `m5a`  

We use the response-delay in the CSV file as the metric for serverless functions' execution time (if the status code is 200). 