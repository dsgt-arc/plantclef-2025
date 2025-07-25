from pyspark.ml.param import Param, Params, TypeConverters


class HasCheckpointPathSAM(Param):
    """
    Mixin for param checkpoint_path: str
    """

    checkpointPathSAM = Param(
        Params._dummy(),
        "checkpointPathSAM",
        "The path to the segment-anything checkpoint weights",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default="facebook/sam-vit-huge",
            doc="The path to the segment-anything checkpoint weights",
        )

    def getCheckpointPathSAM(self) -> str:
        return self.getOrDefault(self.checkpointPathSAM)


class HasCheckpointPathGroundingDINO(Param):
    """
    Mixin for param checkpoint_path_groundingdino: str
    """

    checkpointPathGroundingDINO = Param(
        Params._dummy(),
        "checkpointPathGroundingDINO",
        "The path to the GroundingDINO checkpoint weights",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default="IDEA-Research/grounding-dino-base",
            doc="The path to the GroundingDINO checkpoint weights",
        )

    def getCheckpointPathGroundingDINO(self) -> str:
        return self.getOrDefault(self.checkpointPathGroundingDINO)
