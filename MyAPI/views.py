from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .models import expected_disease
from .serializers import expected_diseaseSerializers
import numpy as np
import pandas as pd
from joblib import load
from django.views.decorators.csrf import csrf_exempt

class Expected_diseaseView(viewsets.ModelViewSet):
  queryset = expected_disease.objects.all()
  serializer_class = expected_diseaseSerializers

@csrf_exempt    
@api_view(["POST"])
def predict_disease(request):
    try:
        mdl = load("./saveModels/disease_model.joblib")
        label_encoder = load('./saveModels/disease_encoder.joblib')
        df1 = pd.read_csv('./Notebooks/Symptom-severity.csv')

        mydata = request.data
        symptoms = mydata.get("symptoms", [])  # list of symptom name strings

        print("Received symptoms:", symptoms)

        # Build 17-length feature vector using CSV (same as web)
        symptom_values = []
        for symptom in symptoms[:17]:
            s_query = df1[df1['Symptom'] == symptom]['weight']
            s_weight = s_query.iloc[0] if not s_query.empty else 0
            symptom_values.append(int(s_weight))

        # Pad to 17
        symptom_values.extend([0] * (17 - len(symptom_values)))
        symptom_values = np.reshape(symptom_values, (1, -1))

        print("Processed Symptoms for Model:", symptom_values)

        y_pred_proba = mdl.predict_proba(symptom_values)
        top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
        top3_proba = y_pred_proba[0][top3_indices]
        top3_diseases = label_encoder.inverse_transform(top3_indices)

        response_data = {
            "top_diseases": [f"{disease}" for disease in top3_diseases],
            "probabilityList": [f"{probability:.2%}" for probability in top3_proba],
        }

        print("Prediction Response:", response_data)
        return JsonResponse(response_data)

    except ValueError as e:
        print("ValueError:", str(e))
        return Response(e.args[0], status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print("Unexpected Error:", str(e))
        return JsonResponse({'error': str(e)}, status=500)


def cxcontact(request):
  return render(request, 'myform/cxform.html')


def formInfo(request):
    df1 = pd.read_csv('./Notebooks/Symptom-severity.csv')
    model = load('./saveModels/disease_model.joblib')

    symptoms = []
    dropdowns = ['dropdown1', 'dropdown2', 'dropdown3', 'dropdown4', 'dropdown5']

    for dropdown in dropdowns:
        symptom = request.GET.get(dropdown, None)
        if symptom:
            s_query = df1[df1['Symptom'] == symptom]['weight']
            s_weight = s_query.iloc[0] if not s_query.empty else 0
        else:
            s_weight = 0
        symptoms.append(s_weight)

    symptoms.extend([0] * (17 - len(symptoms)))

    print([symptoms])
    y_pred_proba = model.predict_proba([symptoms])
    top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top3_proba = y_pred_proba[0][top3_indices]
    label_encoder = load('./saveModels/disease_encoder.joblib')
    top3_diseases = label_encoder.inverse_transform(top3_indices)

    result = [f"{disease}: {probability:.2%}\n" for disease, probability in zip(top3_diseases, top3_proba)]

    return render(request, 'result.html', {'result': result})