from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .models import expected_disease
from .serializers import expected_diseaseSerializers
import pickle
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
from joblib import load
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
class Expected_diseaseView(viewsets.ModelViewSet):
  queryset = expected_disease.objects.all()
  serializer_class = expected_diseaseSerializers

@csrf_exempt    
@api_view(["GET"])
def predict_disease(request):
  try:
    mdl= load("./saveModels/disease_model.joblib")
    mydata=request.data
    symptoms = mydata.get("symptoms", [])
    print(mydata)
    #Preprocess data here if required
    symptom_values = [int(symptom) for symptom in symptoms] 
    symptom_values.extend([0] * (17 - len(symptom_values)))   
    symptom_values = np.reshape(symptom_values, (1, -1))
    print(symptom_values)
    y_pred_proba = mdl.predict_proba(symptom_values)
    top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top3_proba = y_pred_proba[0][top3_indices]
    label_encoder = load('./saveModels/disease_encoder.joblib')
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    response_data = {
            "top_diseases": [f"{disease}: {probability:.2%}" for disease, probability in zip(top3_diseases, top3_proba)],
        }
    return JsonResponse(response_data, safe=False)
  except ValueError as e:
    return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
  
@csrf_exempt    
@api_view(["POST"])  # Handle POST requests
def predict_disease(request):
    try:
        mdl = load("./saveModels/disease_model.joblib")
        label_encoder = load('./saveModels/disease_encoder.joblib')
        mydata = request.data
        symptoms = mydata.get("symptoms", [])

        symptom_weight_mapping = {
          "itching": 1,
          "skin_rash": 3,
          "nodal_skin_eruptions": 4,
          "continuous_sneezing": 4,
          "shivering": 5,
          "chills": 3,
          "joint_pain": 3,
          "stomach_pain": 5,
          "acidity": 3,
          "ulcers_on_tongue": 4,
          "muscle_wasting": 3,
          "vomiting": 5,
          "burning_micturition": 6,
          "spotting_urination": 6,
          "fatigue": 4,
          "weight_gain": 3,
          "anxiety": 4,
          "cold_hands_and_feets": 5,
          "mood_swings": 3,
          "weight_loss": 3,
          "restlessness": 5,
          "lethargy": 2,
          "patches_in_throat": 6,
          "irregular_sugar_level": 5,
          "cough": 4,
          "high_fever": 7,
          "sunken_eyes": 3,
          "breathlessness": 4,
          "sweating": 3,
          "dehydration": 4,
          "indigestion": 5,
          "headache": 3,
          "yellowish_skin": 3,
          "dark_urine": 4,
          "nausea": 5,
          "loss_of_appetite": 4,
          "pain_behind_the_eyes": 4,
          "back_pain": 3,
          "constipation": 4,
          "abdominal_pain": 4,
          "diarrhoea": 6,
          "mild_fever": 5,
          "yellow_urine": 4,
          "yellowing_of_eyes": 4,
          "acute_liver_failure": 6,
          "fluid_overload": 4,
          "swelling_of_stomach": 7,
          "swelled_lymph_nodes": 6,
          "malaise": 6,
          "blurred_and_distorted_vision": 5,
          "phlegm": 5,
          "throat_irritation": 4,
          "redness_of_eyes": 5,
          "sinus_pressure": 4,
          "runny_nose": 5,
          "congestion": 5,
          "chest_pain": 7,
          "weakness_in_limbs": 7,
          "fast_heart_rate": 5,
          "pain_during_bowel_movements": 5,
          "pain_in_anal_region": 6,
          "bloody_stool": 5,
          "irritation_in_anus": 6,
          "neck_pain": 5,
          "dizziness": 4,
          "cramps": 4,
          "bruising": 4,
          "obesity": 4,
          "swollen_legs": 5,
          "swollen_blood_vessels": 5,
          "puffy_face_and_eyes": 5,
          "enlarged_thyroid": 6,
          "brittle_nails": 5,
          "swollen_extremeties": 5,
          "excessive_hunger": 4,
          "extra_marital_contacts": 5,
          "drying_and_tingling_lips": 4,
          "slurred_speech": 4,
          "knee_pain": 3,
          "hip_joint_pain": 2,
          "muscle_weakness": 2,
          "stiff_neck": 4,
          "swelling_joints": 5,
          "movement_stiffness": 5,
          "spinning_movements": 6,
          "loss_of_balance": 4,
          "unsteadiness": 4,
          "weakness_of_one_body_side": 4,
          "loss_of_smell": 3,
          "bladder_discomfort": 4,
          "foul_smell_ofurine": 5,
          "continuous_feel_of_urine": 6,
          "passage_of_gases": 5,
          "internal_itching": 4,
          "toxic_look_(typhos)": 5,
          "depression": 3,
          "irritability": 2,
          "muscle_pain": 2,
          "altered_sensorium": 2,
          "red_spots_over_body": 3,
          "belly_pain": 4,
          "abnormal_menstruation": 6,
          "dischromic_patches": 6,
          "watering_from_eyes": 4,
          "increased_appetite": 5,
          "polyuria": 4,
          "family_history": 5,
          "mucoid_sputum": 4,
          "rusty_sputum": 4,
          "lack_of_concentration": 3,
          "visual_disturbances": 3,
          "receiving_blood_transfusion": 5,
          "receiving_unsterile_injections": 2,
          "coma": 7,
          "stomach_bleeding": 6,
          "distention_of_abdomen": 4,
          "history_of_alcohol_consumption": 5,
          "blood_in_sputum": 5,
          "prominent_veins_on_calf": 6,
          "palpitations": 4,
          "painful_walking": 2,
          "pus_filled_pimples": 2,
          "blackheads": 2,
          "scurring": 2,
          "skin_peeling": 3,
          "silver_like_dusting": 2,
          "small_dents_in_nails": 2,
          "inflammatory_nails": 2,
          "blister": 4,
          "red_sore_around_nose": 2,
          "yellow_crust_ooze": 3,
          "prognosis": 5
      }



        # Convert symptoms to corresponding weights
        symptom_values = [symptom_weight_mapping.get(symptom, 0) for symptom in symptoms]

        # Pad to match model input size (e.g., 17)
        symptom_values.extend([0] * (17 - len(symptom_values)))  
        symptom_values = np.reshape(symptom_values, (1, -1))

        print("Processed Symptoms for Model:", symptom_values)

        # Make prediction
        y_pred_proba = mdl.predict_proba(symptom_values)
        top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
        top3_proba = y_pred_proba[0][top3_indices]
        top3_diseases = label_encoder.inverse_transform(top3_indices)

        # Format response
        response_data = {
            "top_diseases": [
                f"{disease}" for disease in top3_diseases
            ],
            "probabilityList": [
                f"{probability:.2%}" for probability in top3_proba
            ],
        }


        print("Prediction Response Data:", response_data)
        return JsonResponse(response_data)

    except ValueError as e:
        print("Error during Prediction:", str(e))
        return Response(e.args[0], status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print("Unexpected Error:", str(e))
        return JsonResponse({'error': str(e)}, status=500)

  
def cxcontact(request):
  return render(request,'myform/cxform.html')

def formInfo(request):
    import pandas as pd
    df1 = pd.read_csv('./Notebooks/Symptom-severity.csv')
    model = load('./saveModels/disease_model.joblib')
    
    symptoms = []
    dropdowns = ['dropdown1', 'dropdown2', 'dropdown3', 'dropdown4', 'dropdown5']

    for dropdown in dropdowns:
        symptom = request.GET.get(dropdown, None)
        if symptom:  # If symptom is provided
            s_query = df1[df1['Symptom'] == symptom]['weight']
            s_weight = s_query.iloc[0] if not s_query.empty else 0
        else:  # If symptom is not provided
            s_weight = 0
        symptoms.append(s_weight)
    
    # Fill the remaining slots with 0 if fewer than 5 symptoms are provided
    symptoms.extend([0] * (17 - len(symptoms)))

    print([symptoms])
    y_pred_proba = model.predict_proba([symptoms])
    top3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top3_proba = y_pred_proba[0][top3_indices]
    label_encoder = load('./saveModels/disease_encoder.joblib')
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    
    result = [f"{disease}: {probability:.2%}\n" for disease, probability in zip(top3_diseases, top3_proba)]
    
    return render(request, 'result.html', {'result': result})
