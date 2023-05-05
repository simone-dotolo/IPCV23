class Sensor:

    def __init__(self, sensor):

        self.sensor = sensor
        self.ratio = 4

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'WV2') or (sensor == 'WV3'):
            self.kernels = [9, 5, 5]
        elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.kernels = [5, 5, 5]

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.nbands = 4
        elif (sensor == 'WV2') or (sensor == 'WV3'):
            self.nbands = 8
            
        self.nbits = 11