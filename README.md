# 10 year CHD Prediction Model
- A `Logistic Regression` model that outputs the probability of a patient having a future 10 year risk of CHD (Coronary Heart Disease).
- The model has accuracy of `0.8360655737704918` with just the standard threshold of `0.5`
- `2nd degree` polynomial features are used.
- Regularization is implemented with `xi.xj for all i == j` and `lambda = 0.5`

 ## Description of the dataset used
 - 15 feature dataset and after regularization total number of features is 31
 - Each attribute is a potential risk factor. There are both demographic, behavioral and medical risk factors.
 - Demographic:
   - Sex: male or female (Nominal)
   - Age: Age of the patient ;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
  - Behavioral
    - Current Smoker: whether or not the patient is a current smoker (Nominal)
    - Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
  - Medical( history)
     - BP Meds: whether or not the patient was on blood pressure medication (Nominal)
     - Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
     - Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
     - Diabetes: whether or not the patient had diabetes (Nominal)
   - Medical(current)
     - Tot Chol: total cholesterol level (Continuous)
     - Sys BP: systolic blood pressure (Continuous)
     - Dia BP: diastolic blood pressure (Continuous)
     - BMI: Body Mass Index (Continuous)
     - Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
     - Glucose: glucose level (Continuous)
  - Predict variable (desired target)
    - 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)
