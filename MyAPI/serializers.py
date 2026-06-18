from rest_framework import serializers
from .models import expected_disease

class expected_diseaseSerializers(serializers.ModelSerializer):
  class Meta:
    models=expected_disease
    fields='__all__'
