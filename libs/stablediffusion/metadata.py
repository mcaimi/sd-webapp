#!/usr/bin/env python

import json

class GenerationMetadata(object):
    def __init__(self, metadict: dict):
        self.metadata = metadict

        # load generation parameters
        try:
            for k in self.metadata.keys():
                setattr(self, f"{k}", self.metadata.get(k))

            # access generation data specifically
            instance_parms: dict = self.output_parameters.get("instances")[0]

            # set attributes
            for k in instance_parms.keys():
                setattr(self, f"{k}", instance_parms.get(k))
        except Exception as e:
            raise Exception(f"Cannot load Metadata from Dictionary: {e}")