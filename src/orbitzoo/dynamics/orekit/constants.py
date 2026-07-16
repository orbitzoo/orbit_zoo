from org.orekit.frames import FramesFactory, LOFType
from org.orekit.utils import IERSConventions, Constants
from org.orekit.attitudes import LofOffset
from org.orekit.time import TimeScalesFactory

EARTH_FLATTENING = Constants.WGS84_EARTH_FLATTENING
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
MOON_EQUATORIAL_RADIUS = Constants.MOON_EQUATORIAL_RADIUS
MU = Constants.WGS84_EARTH_MU
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
UTC = TimeScalesFactory.getUTC()
INERTIAL_FRAME = FramesFactory.getEME2000()
ATTITUDE = LofOffset(INERTIAL_FRAME, LOFType.LVLH)
MEME = FramesFactory.getMOD(IERSConventions.IERS_2010)