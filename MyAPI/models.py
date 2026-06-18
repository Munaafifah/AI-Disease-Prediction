from django.db import models

# Create your models here.
class expected_disease(models.Model):
  SYMPTOMCHOICES = (
    (5,'batuk'),
    (7,'wtf'),
  )
  
  symptom1 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom2 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom3 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom4 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom5 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom6 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom7 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom8 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom9 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom10 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom11 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom12 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom13 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom14 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom15 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom16 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  symptom17 = models.CharField(max_length=50,choices=SYMPTOMCHOICES)
  
  def __str__(self):
    return '{},{}'.format(self.symptom1,self.symptom2)