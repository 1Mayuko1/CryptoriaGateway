from django.urls import path
from .views import create_chart
from .views import get_historical_data
from .views import get_symbols

urlpatterns = [
    path('create/<str:name>/<str:start_time>/<str:end_time>', create_chart, name='chartCreator'),
    path('get_historical_data/<str:name>/<int:count>/<str:start_time>/<str:end_time>/',
         get_historical_data, name='chartCreator'),
    path('get_historical_data/<str:name>/<int:count>/<str:start_time>/',
         get_historical_data, name='chartCreator'),
    path('get_symbols', get_symbols, name='chartCreator'),
]
