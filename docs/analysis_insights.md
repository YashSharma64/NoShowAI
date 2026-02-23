# NoShowAI – Analysis & Insights

## 1. Overview

Total Records: 71,959
No-show Rate: ~28.5%

The dataset shows a moderate class imbalance but remains suitable for predictive modeling and behavioral analysis.

---

## 2. Key Findings

### 2.1 LeadTime (Waiting Period)

LeadTime represents the number of days between booking and the appointment date.
    - Longer LeadTime significantly increases the probability of no-show.
    - Patients scheduled far in advance are more likely to forget or deprioritize the visit.
    - Reduced urgency over time contributes to missed appointments.

Observation: LeadTime is one of the strongest predictive features in the model.

### 2.2 Age

Age plays a major role in attendance behavior.
    - Younger patients show a higher tendency to miss appointments.
    - Older patients generally attend more consistently.

Possible Reasons:
    - Work or academic commitments
    - Lower perceived health urgency
    - Schedule variability

### 2.3 Appointment Weekday

Attendance patterns vary depending on the day of the week.
    - Certain weekdays show higher no-show rates.
    - Work schedules and lifestyle patterns likely influence attendance.

This indicates that scheduling strategies can be optimized based on weekday trends.

### 2.4 SMS Reminder Impact

Patients who receive SMS reminders demonstrate improved attendance rates.

This confirms that structured communication systems help reduce no-show probability.

---

## 3. Model-Based Feature Importance

A Random Forest classifier was trained to evaluate predictive importance.

Top contributing features:
    1. Age
    2. LeadTime
    3. AppointmentWeekday

The model confirms the patterns identified during exploratory analysis.

---

## 4. Business Recommendations

Based on the analysis, the following actions may reduce no-show rates:
- Reduce LeadTime between booking and appointment
- Strengthen automated SMS reminder systems
- Use predictive scoring to identify high-risk patients
- Optimize scheduling on high no-show weekdays
- Provide targeted reminders for younger patients

---

## 5. Conclusion

The analysis demonstrates that demographic factors (Age) and temporal factors (LeadTime, AppointmentWeekday) significantly influence appointment attendance behavior.

By combining predictive modeling with operational improvements, healthcare providers can reduce no-show rates and improve resource utilization.