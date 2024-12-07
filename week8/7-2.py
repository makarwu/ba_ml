import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Provided data
data = [
    (1, 0.991, '+'), (2, 0.977, '+'), (3, 0.973, '+'),
    (4, 0.945, '+'), (5, 0.918, '+'), (6, 0.915, '-'),
    (7, 0.906, '+'), (8, 0.889, '-'), (9, 0.873, '+'),
    (10, 0.871, '+'), (11, 0.869, '-'), (12, 0.866,'-'),
    (13 ,0.862,'+'), (14 ,0.852,'-'), (15 ,0.837,'+'),
    (16 ,0.831,'-'), (17 ,0.829,'-'), (18 ,0.811,'-'),
    (19 ,0.787,'-'), (20 ,0.779,'-')
]

# Extract probabilities and classes
probabilities = np.array([x[1] for x in data])
classes = np.array([1 if x[2] == '+' else 0 for x in data])

# Sort by probability
sorted_indices = np.argsort(probabilities)[::-1]
sorted_probabilities = probabilities[sorted_indices]
sorted_classes = classes[sorted_indices]

# Total positives
total_positives = np.sum(sorted_classes)

# Gain Curve Calculation
cumulative_positives = np.cumsum(sorted_classes)
gain_curve = cumulative_positives / total_positives

# Lift Curve Calculation
lift_curve = gain_curve / ((np.arange(1,len(gain_curve)+1)) / len(classes))

# ROC Curve Calculation
fpr,tpr,_ = roc_curve(classes , probabilities)

# Plotting Gain Curve
plt.figure(figsize=(12,8))

plt.subplot(3 ,1 ,1)
plt.plot(np.arange(1,len(gain_curve)+1)/len(gain_curve), gain_curve)
plt.axhline(y=total_positives/len(classes), color='r', linestyle='--', label='Random Guessing')
plt.axvline(x=0.87 ,color='g', linestyle='--', label='Cutoff at 87%')
plt.title('Gain Curve')
plt.xlabel('Percentage of Instances')
plt.ylabel('Gain')
plt.legend()

# Plotting Lift Curve
plt.subplot(3 ,1 ,2)
plt.plot(np.arange(1,len(lift_curve)+1)/len(lift_curve), lift_curve)
plt.axhline(y=1 ,color='r', linestyle='--', label='Random Guessing Lift')
plt.axvline(x=0.87 ,color='g', linestyle='--', label='Cutoff at 87%')
plt.title('Lift Curve')
plt.xlabel('Percentage of Instances')
plt.ylabel('Lift')
plt.legend()

# Plotting ROC Curve
plt.subplot(3 ,1 ,3)
plt.plot(fpr ,tpr)
plt.plot([0 ,1] ,[0 ,1] ,'r--') # Random guessing line
plt.axvline(x=1-0.87,ymin=0,color='g', linestyle='--', label='Cutoff at 87%')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.tight_layout()
plt.show()