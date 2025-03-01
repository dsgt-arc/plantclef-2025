path=$1

python <<EOF
from plantclef.spark import get_spark
df = get_spark().read.parquet("$path")
df.printSchema()
df.show(n=1, vertical=True, truncate=100)
print("count", df.count())
EOF