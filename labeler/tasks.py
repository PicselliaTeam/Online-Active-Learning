import datetime
import time
import os
from celery import Celery

celery_worker = Celery('worker', broker='redis://127.0.0.1:6382/0')

@celery_worker.task
def hello():
    time.sleep(10)
    with open ('hellos.txt', 'a') as hellofile:
        hellofile.write('Hello {}\n'.format(datetime.datetime.now()))


@celery_worker.task(name="annotate")
def annotate_task():
    labeler.annotate()
     