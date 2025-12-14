from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),

    # User Authentication
    path('login/', views.UserLogin, name='UserLogin'),
    path('login-action/', views.UserLoginAction, name='UserLoginAction'),
    path('register/', views.Register, name='Register'),
    path('register-action/', views.RegisterAction, name='RegisterAction'),

    # Dataset Processing
    path('load-dataset/', views.LoadDatasetAction, name='LoadDatasetAction'),
    path('process-data/', views.ProcessData, name='ProcessData'),

    # Machine Learning / Prediction
    path('run-ml/', views.RunML, name='RunML'),
    path('predict/', views.Predict, name='Predict'),
    path('predict-action/', views.PredictAction, name='PredictAction'),
]
