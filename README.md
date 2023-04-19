# Online Trajectory Anomaly Detection Based on Intention Orientation
This repo contains the implementation of algorithm IO-TAD from the paper Online Trajectory Anomaly Detection Based on Intention Orientation. It includes code for data processing, Inverse Reinforcemnet Learning(IRL) and online anomaly detection.
### Dependencies
- Python 3.8
- Tensorflow
- Scikit-learn

### Data Processing
We will transfer the orginal GPS trajectories to grid trajectories with state-action pairs, and derive the destination-based trajectory clusters. For synthetic anomalies, we provide the code of anomaly generation for three types of anomaly: Wrong-destination anomalies, detour anomalies and random walk anoamlies.
### Inverse Reinforcement Learning
The code implements deep maximum entropy IRL.
### Online Detection
To implement IO-TAD on Chengdu dataset, please run main.py from the file online detection.
### Authors
Chen Wang, Sarah Erfani, Tansu Alpcan, Christopher Leckie
