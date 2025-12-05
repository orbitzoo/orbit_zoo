from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.models.earth.atmosphere import NRLMSISE00, data as atmosphere_data
from org.orekit.forces.drag import IsotropicDrag, DragForce as Drag

from dynamics.constants import MOON_EQUATORIAL_RADIUS

class ThirdBodyForce(ThirdBodyAttraction):
    def __init__(self, name: str):
        allowed_bodies = {'SOLAR_SYSTEM_BARYCENTER', 'SUN', 'MERCURY', 'VENUS', 'EARTH_MOON', 'EARTH', 'MOON', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO'}
        if name not in allowed_bodies:
            raise ValueError(f"Invalid third body with name '{name}'. Allowed names are: {', '.join(allowed_bodies)}")
        third_body = CelestialBodyFactory.getBody(name)
        super().__init__(third_body)

class SolarRadiationForce(SolarRadiationPressure):
    def __init__(self, earth: OneAxisEllipsoid, surface_area: float, reflection_coefficient: float):
        body = IsotropicRadiationSingleCoefficient(surface_area, reflection_coefficient)
        super().__init__(CelestialBodyFactory.getSun(), earth, body)
        self.addOccultingBody(CelestialBodyFactory.getMoon(), MOON_EQUATORIAL_RADIUS)

class DragForce(Drag):
    def __init__(self, earth: OneAxisEllipsoid, surface_area: float, drag_coefficient: float):
        weather_data = atmosphere_data.CssiSpaceWeatherData(atmosphere_data.CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
        atmosphere = NRLMSISE00(weather_data, CelestialBodyFactory.getSun(), earth)
        body = IsotropicDrag(surface_area, drag_coefficient)
        super().__init__(atmosphere, body)