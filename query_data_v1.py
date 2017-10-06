import googlemaps
import time
from datetime import datetime
import json
import os

gmaps = googlemaps.Client(key='Enter your API key here')

# Replace with your favorite addresses here
addresses = ['157 Berkeley St, Boston, MA',
             '351 Washington St, Brighton, MA',
             '1245 Worcester St, Natick, MA',
             '466 Main St, Woburn, MA',
             '581 Massachusetts Ave, Cambridge, MA',
             '219 Quincy Ave, Quincy, MA',
             '1004-1006 Beacon St, Brookline, MA',
             '290 Main St, Malden, MA',
             '300 Hanover St, Boston, MA',
             '388 S Main St, Sharon, MA']


place_ids = []

tic = time.time()

for address in addresses:
    
    geocode_result = gmaps.geocode(address)
    
    loc = geocode_result[0]['geometry']['location']
    
    data = gmaps.places('restaurant',(loc['lat'], loc['lng']))
    
    place_ids = place_ids + [i['place_id'] for i in data['results']]
    
    time.sleep(2)
    
    for j in range(4):
        
        if 'next_page_token' not in data.keys(): break
        
        data = gmaps.places('restaurant',(loc['lat'], loc['lng']),page_token=data['next_page_token'])
        
        place_ids = place_ids + [i['place_id'] for i in data['results']]
        
        time.sleep(2)

toc = time.time()
toc - tic
# 130 seconds

len(place_ids)
# 660

len(set(place_ids))
# 586

place_ids = list(set(place_ids))


tic = time.time()
for i, place_id in enumerate(place_ids):
    
    data = gmaps.place(place_id)
    
    with open('raw/' + str(i) + '.json', 'w') as outfile:
        json.dump(data, outfile)

    time.sleep(1)

toc = time.time()
toc - tic

# timeout at 472
# place_id: ChIJwZlVC1R344kRFd80FfZkL_4
