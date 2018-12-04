import reverse_geocoder as rg
import io
import json
# pip install reverse_geocoder

# lat,long?
coordinates = (40.77010669, -73.88530464), (40.77737697, -73.88077509)
results = rg.search(coordinates) # default mode = 2
list_dict = [dict(result) for result in results]

new_dict = {str((item['lat'], item['lon'])): item for item in list_dict}

print(new_dict)

with io.open('data.json', 'w') as outfile:
    json.dump(new_dict, outfile)
