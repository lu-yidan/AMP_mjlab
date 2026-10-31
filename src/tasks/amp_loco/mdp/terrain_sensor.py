"""Invalidate ray data after sensing, even if rewards read it earlier this step."""
from dataclasses import dataclass
from mjlab.sensor import RayCastSensor, RayCastSensorCfg


class FreshTerrainRayCastSensor(RayCastSensor):
    def postprocess_rays(self):
        super().postprocess_rays()
        # mjlab caches a RayCastData object; postprocess replaces its tensors.
        # Ground-relative rewards can fill that cache before the new scan exists.
        self._invalidate_cache()


@dataclass
class FreshTerrainRayCastSensorCfg(RayCastSensorCfg):
    def build(self):
        return FreshTerrainRayCastSensor(self)
