import faiss
import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType, IntegerType

from plantclef.retrieval.query.index_setup import setup_index


class HasIndexPath(Param):
    """
    Mixin for param index_path: str
    """

    indexPath = Param(
        Params._dummy(),
        "indexPath",
        "The path to the FAISS index",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super(HasIndexPath, self).__init__()
        self._setDefault(indexPath=None)

    def getIndexPath(self) -> str:
        return self.getOrDefault(self.indexPath)


class HasIndexName(Param):
    """
    Mixin for param index_name: str
    """

    indexName = Param(
        Params._dummy(),
        "indexName",
        "The name of the FAISS index",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__(
            default="embeddings",
            doc="The name of the FAISS index",
        )

    def getIndexName(self) -> str:
        return self.getOrDefault(self.indexName)


class HasK(Param):
    """
    Mixin for param k: int
    """

    k = Param(
        Params._dummy(),
        "k",
        "The number of nearest neighbors to return",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self):
        super().__init__(
            default=10,
            doc="The number of nearest neighbors to return",
        )

    def getK(self) -> int:
        return self.getOrDefault(self.k)


class FaissNearestNeighbors(
    Transformer, 
    HasInputCol, 
    HasIndexPath, 
    HasIndexName, 
    HasK,
    DefaultParamsReadable, 
    DefaultParamsWritable
):
    """
    Wrapper for FAISS nearest neighbors search.
    """
    
    def __init__(
        self,
        input_col: str = "input",
        index_path: str = None,
        index_name: str = "embeddings",
        k: int = 10,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            indexPath=index_path,
            indexName=index_name,
            k=k,
        )
        if self.getIndexPath() is None:
            index_path = setup_index(scratch_model=True, index_name=self.getIndexName())
            self._set(indexPath=index_path)
        # self.index = faiss.read_index(str(self.getIndexPath()))
        
    # def _transform(self, df: DataFrame) -> DataFrame:
        
    #     index = faiss.read_index(str(self.getIndexPath()))
        
    #     def process_partition(partition_iter):
    #         batch = []
    #         rows = []
    #         for row in partition_iter:
    #             batch.append(row[self.getInputCol()])
    #             rows.append(row)
            
    #         embeddings = np.stack(batch).astype("float32")
    #         faiss.normalize_L2(embeddings)
    #         distances, ids = index.search(embeddings, self.getK())
            
    #         results = []
    #         for i, row in enumerate(rows):
    #             new_row = row.asDict()
    #             new_row["distances"] = distances[i].tolist()
    #             new_row["ids"] = ids[i].tolist()
    #             results.append(new_row)
                
    #         return results
        
        # out_schema = df.schema.add("distances", ArrayType(FloatType())).add("ids", ArrayType(IntegerType()))
        # return df.rdd.mapPartitions(process_partition).toDF(out_schema)
        
        # return df.mapInPandas(
        #     lambda iterator: process_partition(iterator), 
        #     schema=df.schema.add("distances", ArrayType(FloatType()))
        #                     .add("ids", ArrayType(IntegerType()))
        # )
