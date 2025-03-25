# State and Parameter Estimation for Model Predictive Control of Buildings

## 📌 Overview
This project focuses on **batch estimation** techniques for estimating both states and parameters in building thermal models, which are crucial for effective Model Predictive Control (MPC). Using grey-box modeling and non-linear optimization, we explored multiple RC-network-based thermal models (1-state, 2-state, and 4-state) to evaluate how different modeling complexities impact the prediction and control of indoor environments.

This project was developed as part of the **Advanced Controls** course at **Washington State University**.

---

## Tools & Technologies
- **Programming Language**: Python
- **Libraries Used**: NumPy, Matplotlib, CasADi
- **Estimation Techniques**:  
  - Batch Estimation  
  - State Augmentation  
  - Maximum Likelihood Estimation  
  - Nonlinear Least Squares  
- **Model Types**:
  - 1-State, 2-State, and 4-State RC Thermal Models  
  - Simple Pendulum for validation and control testing

---

##  Methodology

### 🔸 Grey-Box Thermal Modeling
- Utilized RC-network-based ODEs to simulate thermal dynamics in buildings
- Developed multiple models (1-state to 4-state) to balance accuracy and complexity

### 🔸 Batch Estimation
- Formulated as a nonlinear optimization problem using CasADi
- Incorporated system and measurement noise using Gaussian assumptions
- Explored the use of physical constraints and prior knowledge (MAP estimation)

### 🔸 Parameter Estimation
- Applied both **Bayesian filtering** (via state augmentation) and **least squares optimization**
- Implemented Maximum Likelihood Estimation using Kalman filtering for dynamic systems

---

##  Results

- Developed and tested thermal models using synthetic and prototype data
- Demonstrated the influence of model complexity on estimation accuracy
- Validated the effectiveness of batch estimation on a **Simple Pendulum System** with known dynamics

---


##  Documentation

-  [Report](./docs/report.pdf)  
-  [Presentation Script](./docs/AI%20Script.docx)  
-  [Working Notes](./docs/Batch%20Estimation%20Work.docx)

---

##  My Role

I contributed to:
- Building and discretizing the thermal models
- Implementing optimization logic using CasADi
- Developing batch estimation code for various models
- Analyzing results and preparing technical documentation

---

##  License
This project is intended for academic and demonstration purposes. You may add a license such as MIT for open-source sharing.

---

##  Contact
Feel free to reach out via [LinkedIn](https://linkedin.com/in/chinmaychabbi) or GitHub for any questions or collaboration opportunities.

