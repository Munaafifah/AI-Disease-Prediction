from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register("MyAPI", views.Expected_diseaseView)
urlpatterns = [
    path('api/', include(router.urls)),
    path('status/',views.predict_disease,name='Predict Disease'),
    path('form/',views.cxcontact,name='Common Disease Prediction'),
    path('form/result', views.formInfo, name='Prediction Result')
]
