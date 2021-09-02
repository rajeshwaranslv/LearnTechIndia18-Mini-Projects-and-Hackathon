# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 08:22:15 2021

@author: Sanju Pranava
"""

import requests

url = "https://zyanyatech1-license-plate-recognition-v1.p.rapidapi.com/recognize_url"

querystring = {"image_url":"http://eslamoda.com/wp-content/uploads/sites/2/2014/11/america-carro-600x600.jpg"}

headers = {
    'x-rapidapi-key': "62c69479c1msh7557e81fc0cc36dp15127djsn652813f9417a",
    'x-rapidapi-host': "zyanyatech1-license-plate-recognition-v1.p.rapidapi.com"
    }

response = requests.request("POST", url, headers=headers, params=querystring)

print(response.text)