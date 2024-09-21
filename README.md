# Aiming for Relevance: Evaltion Metrics for Vital Signs Prediction Models that Considers Clinical Relevance

_Aiming for Relevance_ [1] introduces new metrics for evaluating machine learning models in vital sign prediction. While classification tasks often use a variety of metrics like precision and recall to capture different aspects of performance, time series prediction typically relies on metrics like RMSE and MAPE, which tend to focus on overall accuracy. However, a prediction that is generally accurate, but misses clinically important events could have grave implication; for example, a blood pressure prediction that is accurate for a long periods of time in which the patient is stable, completely miss a sudden drop in blood pressure. Standard metrics often miss these clinically important details such as deviations from normal ranges, evolving trends or shifts in trends. To address this, [1] propose utility-based metrics, informed by clinician interviews, that complement RMSE. Though developed for vital sign prediction, these metrics could also benefit other time series prediction tasks.

The clinical utility of vital sign predictions is closely tied to both severity and surprise—how extreme or unexpected a prediction is. For instance, predicting a heart rate of 180 BPM signals immediate danger. To capture this, utility curves were formulated to quantify the cost of predictions that deviate from clinically normal ranges or expected trends, based on clinicians' inputs [2].



The full paper is available available [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11141809/).

# Utility Costs 

Each evaluation metric correspond to a different utility cost according to the aspect it conveys.

### Normal Range Utility Cost
For normal ranges, we model the importance of predictions using a two-sided sigmoid curve, which increases in significance as values move further from clinically defined normal bounds. The normal range utility cost $U_{r}$ is calculated as:

$U_r(y_t, \hat{y}_t) = \left| \left( \max\left(\frac{L}{1 + e^{k_h \cdot (y_t - h)}}, 0\right) + \max\left(\frac{L}{1 + e^{k_l \cdot (y_t - l)}}, 0\right) \right) - \left( \max\left(\frac{L}{1 + e^{k_h \cdot (\hat{y}_t - h)}}, 0\right) + \max\left(\frac{L}{1 + e^{k_l \cdot (\hat{y}_t - l)}}, 0\right) \right) \right|$

$h, l, k_{h},k_{l} L$ are parameters controlling the high and low thresholds and sigmoid steepness for clinical importance. The utility cost rises when the prediction falls outside the clinically normal range.

### Trend Utility Cost:
For trends, we calculate the difference between the predicted and actual slopes. The trend utility cost $U_{t}$ is designed to penalize sharper deviations, especially in critical cases like a drop in blood pressure. It is computed as:
$U_{tr}(Y_{t-n:t+m}, \hat{Y}_{t-n:t+m}) = \max(\hat{Y}_{t-n:t+m} - Y_{t-n:t+m}, 0)^2 \cdot w_l + \max(Y_{t-n:t+m} - \hat{Y}_{t-n:t+m}, 0)^2 \cdot w_h$

### Trend Deviation Utility Cost:
The trend deviation cost $U_{td}$ measures how surprising the difference between the expected and actual trends is. This is computed by comparing the trend prediction error to the deviation from an expected trend:

$U_{td}(Y_{t-n:t+m}, \hat{Y}_{t-n:t+m}, y_t, \hat{y}_t) = \left( Y'_{t-n:t} - Y_{t-n:t+m} \right)^2 \cdot |y_t - \hat{y}_t|$

Here, $Y'_{t-n:t}$ represents the expected trend over the previous $n$ steps, and $Y_{t-n:t+m}$ is the actual trend. This utility cost emphasizes how unexpected a trend is based on prior information.

Each of these utility costs provides a different angle on how clinically useful a prediction is, allowing models to focus on events that are more relevant to real-world clinical decision-making.

# Code


# references 

[1] Eini-Porat, B.; Eytan, D.; and Shalit, U. 2024. Aiming for Relevance. AMIA Summits on Translational Science Proceedings, 2024: 145

[2] Eini-Porat, B., Amir, O., Eytan, D., & Shalit, U. (2022). Tell me something interesting: Clinical utility of machine learning prediction models in the ICU. Journal of Biomedical Informatics, 132, 104107.‏
