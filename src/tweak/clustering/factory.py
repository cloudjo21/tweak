from tweak.clustering.hac import HAC


class NotSupportedNncException(Exception):
    pass


class NncFactory:
    @classmethod
    def create(cls, nnc_type, dist_threshold):
        # TODO define Enum types for nnc_type
        if nnc_type == "HAC":
            return HAC(dist_threshold)
        else:
            raise NotSupportedNncException(f"Not Supported NNC_TYPE: {nnc_type}")
