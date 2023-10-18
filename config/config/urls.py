from django.urls import path, include


urlpatterns = [
    path('api/', include([
        # Chart API
        path('chart/', include('chartCreator.urls'), name='chartCreator'),
        # Data API
        # path('data/', include('chartCreator.urls'), name='chartCreator'),
    ]))
]