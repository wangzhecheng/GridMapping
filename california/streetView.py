import urllib.request
import urllib
import json


class StreetView():
    """Google Street View API application class"""

    def __init__(self):
        self.api_key_list = [
            '***************************************', # Your API key 1
            '***************************************', # Your API key 2
            '***************************************', # Your API key 3
        ]
        self.key_rank = 0
        self.API_KEY = self.api_key_list[self.key_rank]

    def getMetaStatus(self, location, heading='0', pitch='0', fov='90',
                      radius='50', returnMeta=False):
        # retrieve the meta data through meta API. The returned variable is a dictionary.
        # If the image exists at that point, meta['status'] = 'OK', otherwise: meta['status'] = 'ZERO_RESULTS'
        # For every coordinates, please run this to check existence before downloading image!!
        self.setParameters(location=location, heading=heading,
                           pitch=pitch, fov=fov, radius=radius)
        # print(self.parameters)
        meta_url = 'https://maps.googleapis.com/maps/api/streetview/metadata?'+self.parameters
        response = urllib.request.urlopen(meta_url)
        meta = response.read()
        meta = json.loads(meta)
        if (returnMeta):
            self.nextKey()
            return meta['status'], meta
        else:
            return meta['status']

    def getStreetView(self, location, filepath='dafault.jpg', heading='0',  pitch='0', fov='90', radius='50', uncheck=True):
        # download images from image_url to local path
        # You need to specify "size", "location", "fov", "heading", "pitch", see details in Google API's documentation
        # in Python 3, you may use urllib.request.urlretrieve instead of urllib.urlretrieve
        # uncheck==False: download without check meta
        # filepath : image save path
        if(uncheck):
            status = self.getMetaStatus(
                location=location, heading=heading, pitch=pitch, fov=fov, radius=radius)
        else:
            self.setParameters(location=location, heading=heading,
                               pitch=pitch, fov=fov, radius=radius)
            status = 'OK'

        if(status == 'OK'):
            image_url = 'https://maps.googleapis.com/maps/api/streetview?' + self.parameters
            local_path = filepath
            urllib.request.urlretrieve(image_url, local_path)
            self.nextKey()
            # print('Success!')
            return True
        elif (status == 'ZERO_RESULTS'):
            # not found
            # print('ZERO_RESULTS')
            return False
        else:
            print('Error:'+status)
            return False

    def nextKey(self):
        self.key_rank += 1
        if (self.key_rank >= len(self.api_key_list)):
            self.key_rank = 0
        self.API_KEY = self.api_key_list[self.key_rank]

    def setParameters(self, location, heading, pitch, fov, radius):
        size = ['640', '640']
        #latitude, longitude
        location = [str(location[0]), str(location[1])]
        self.parameters = 'size=' + size[0] + 'x' + size[1] + '&location=' + location[0] + ',' + location[1] + \
            '&fov=' + fov + '&heading=' + heading + '&pitch=' + pitch + \
            '&radius=' + radius + '&key=' + self.API_KEY
