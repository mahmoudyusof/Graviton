import json
import pandas as pd
import numpy as np

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

spark = sparknlp.start()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("documents")

t5 = T5Transformer() \
      .pretrained("t5_small", 'en') \

@app.get('/summaries')
async def summaries(q:str):
   
    t5.setTask("summarize:")\
      .setMaxOutputLength(200)\
      .setInputCols(["documents"]) \
      .setOutputCol("summaries")

    pipeline = Pipeline(stages=[
        document_assembler, t5
    ])  
    df = spark.createDataFrame([[q]]).toDF('text')


    model = pipeline.fit(df)
    annotated_df = model.transform(df)
    return {'result': annotated_df.select(['summaries'])[0].result}


@app.get('/questionanswering')
async def summaries(q:str):
   
    t5.setTask("SQuAD:")\
      .setMaxOutputLength(200)\
      .setInputCols(["documents"]) \
      .setOutputCol("summaries")

    pipeline = Pipeline(stages=[
        document_assembler, t5
    ])  
    df = spark.createDataFrame([[q]]).toDF('text')


    model = pipeline.fit(df)
    annotated_df = model.transform(df)
    return {'result': annotated_df.select(['summaries'])[0].result}


@app.get('/translation')
async def summaries(q:str):
   
    t5.setTask("WMT1:")\
      .setMaxOutputLength(200)\
      .setInputCols(["documents"]) \
      .setOutputCol("summaries")

    pipeline = Pipeline(stages=[
        document_assembler, t5
    ])  
    df = spark.createDataFrame([[q]]).toDF('text')


    model = pipeline.fit(df)
    annotated_df = model.transform(df)
    return {'result': annotated_df.select(['summaries'])[0].result}


@app.get('/sentimentanalysis')
async def summaries(q:str):
    
    t5.setTask("SST2:")\
      .setMaxOutputLength(200)\
      .setInputCols(["documents"]) \
      .setOutputCol("summaries")

    pipeline = Pipeline(stages=[
        document_assembler, t5
    ])  
    df = spark.createDataFrame([[q]]).toDF('text')


    model = pipeline.fit(df)
    annotated_df = model.transform(df)
    return {'result': annotated_df.select(['summaries'])[0].result}
