# Aiming for Relevance: Evaltion Metrics for Vital Signs Prediction Models that Considers Clinical Relevance

_Aiming for Relevance_ (Eini-Porat, Eytan, and Shalit, 2024) introduces new metrics for evaluating machine learning models in vital sign prediction. While classification tasks often use a variety of metrics like precision and recall to capture different aspects of performance, time series prediction typically relies on metrics like RMSE and MAPE, which tend to focus on overall accuracy. However, a prediction that is generally accurate, but misses clinically important events could have grave implication; for example, a blood pressure prediction that is accurate for a long periods of time in which the patient is stable, completely miss a sudden drop in blood pressure. Standard metrics often miss these clinically important details such as deviations from normal ranges, evolving trends or shifts in trends. To address this, (Eini-Porat, Eytan, and Shalit, 2024) propose utility-based metrics, informed by clinician interviews, that complement RMSE. Though developed for vital sign prediction, these metrics could also benefit other time series prediction tasks.

The full paper is available available [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11141809/).

# Utility Costs 

# Code


# Refrence 

[1] Eini-Porat, B.; Eytan, D.; and Shalit, U. 2024. Aiming for
Relevance. AMIA Summits on Translational Science Proceedings, 2024: 145
