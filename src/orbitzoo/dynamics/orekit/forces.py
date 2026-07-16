from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.models.earth.atmosphere import NRLMSISE00, data as atmosphere_data
from org.orekit.forces.drag import IsotropicDrag, DragForce as Drag

from orbitzoo.dynamics.constants import MOON_EQUATORIAL_RADIUS

# These are factory functions (not subclasses): JPype cannot extend Java classes,
# so instead of subclassing the Orekit force models we build and return the real
# Java instance. Behaviour is identical — nothing was overridden, they only
# wrapped constructor logic.

def ThirdBodyForce(name: str) -> ThirdBodyAttraction:
    allowed_bodies = {'SOLAR_SYSTEM_BARYCENTER', 'SUN', 'MERCURY', 'VENUS', 'EARTH_MOON', 'EARTH', 'MOON', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO'}
    if name not in allowed_bodies:
        raise ValueError(f"Invalid third body with name '{name}'. Allowed names are: {', '.join(allowed_bodies)}")
    third_body = CelestialBodyFactory.getBody(name)
    return ThirdBodyAttraction(third_body)

def SolarRadiationForce(earth: OneAxisEllipsoid, surface_area: float, reflection_coefficient: float) -> SolarRadiationPressure:
    body = IsotropicRadiationSingleCoefficient(surface_area, reflection_coefficient)
    force = SolarRadiationPressure(CelestialBodyFactory.getSun(), earth, body)
    force.addOccultingBody(CelestialBodyFactory.getMoon(), MOON_EQUATORIAL_RADIUS)
    return force

def DragForce(earth: OneAxisEllipsoid, surface_area: float, drag_coefficient: float) -> Drag:
    weather_data = atmosphere_data.CssiSpaceWeatherData(atmosphere_data.CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
    atmosphere = NRLMSISE00(weather_data, CelestialBodyFactory.getSun(), earth)
    body = IsotropicDrag(surface_area, drag_coefficient)
    return Drag(atmosphere, body)